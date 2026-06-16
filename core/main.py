"""DeepFM 训练、验证、测试与导出入口。

示例：
    python -m core.main --mode train
    python -m core.main --mode test --checkpoint checkpoints/best.pt
    python -m core.main --mode export --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.amp import GradScaler, autocast

from core.data import build_dataloaders
from core.model import build_model

MIN_PYTHON = (3, 12)


def validate_runtime() -> None:
    """检查运行时 Python 版本。

    当前工程按 Python 3.12+ 维护，避免生产环境和依赖声明漂移。
    """

    if sys.version_info < MIN_PYTHON:
        version = ".".join(map(str, MIN_PYTHON))
        current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        raise RuntimeError(f"当前 Python={current}，本工程要求 Python>={version}")


def load_config(path: str | Path) -> Dict:
    """读取 YAML/JSON 配置。"""

    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def deep_merge_config(base: Dict, override: Dict) -> Dict:
    """递归合并配置，override 优先。"""

    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_runtime_config(
    common_config_path: str | Path,
    model_config_path: str | Path,
    override_config_path: Optional[str | Path] = None,
) -> Dict:
    """加载 common + model + 可选覆盖配置。

    common.yaml 放数据、训练、导出、debug 等工程配置；model.yaml 只放模型结构。
    `--config` 保留为兼容和实验覆盖入口，会在最后合并。
    """

    config = deep_merge_config(load_config(common_config_path), load_config(model_config_path))
    if override_config_path:
        config = deep_merge_config(config, load_config(override_config_path))
    return config


def set_seed(seed: int) -> None:
    """固定随机种子，提升实验可复现性。"""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(value: Any, device: torch.device) -> Any:
    """递归移动 batch 中的 tensor；字符串/list 等元信息原样保留。"""

    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
        return [move_to_device(item, device) for item in value]
    return value


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: move_to_device(value, device) for key, value in batch.items()}


def build_summary_writer(config: Dict):
    """按配置创建 TensorBoard SummaryWriter。

    tensorboard 是可选依赖；未安装时不阻断训练，只打印提示。
    """

    tb_cfg = config.get("debug", {}).get("tensorboard", {})
    if not bool(tb_cfg.get("enabled", False)):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("TensorBoard 未安装，已跳过日志记录。可执行: pip install tensorboard")
        return None

    log_dir = tb_cfg.get("log_dir", "runs/model")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志目录: {log_dir}")
    return writer


def safe_add_scalar(writer: Any, tag: str, value: Any, step: int) -> None:
    """写 scalar，自动跳过 NaN/None，避免 TensorBoard 曲线里出现脏点。"""

    if writer is None or value is None:
        return
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return
    writer.add_scalar(tag, value, step)


def log_metrics(writer: Any, prefix: str, metrics: Dict[str, float], step: int) -> None:
    """写一组指标到 TensorBoard。"""

    if writer is None:
        return
    for name, value in metrics.items():
        safe_add_scalar(writer, f"{prefix}/{name}", value, step)


def log_model_debug_info(writer: Any, model: nn.Module, step: int, log_histograms: bool = False) -> None:
    """记录参数/梯度范数，以及可选直方图。

    直方图写入较重，建议只在小模型或低频调试时开启。
    """

    if writer is None:
        return

    model = unwrap_model(model)
    total_param_norm_sq = 0.0
    total_grad_norm_sq = 0.0
    for name, param in model.named_parameters():
        param_norm = param.detach().float().norm(2).item()
        total_param_norm_sq += param_norm * param_norm
        safe_add_scalar(writer, f"debug/param_norm/{name}", param_norm, step)

        if param.grad is not None:
            grad_norm = param.grad.detach().float().norm(2).item()
            total_grad_norm_sq += grad_norm * grad_norm
            safe_add_scalar(writer, f"debug/grad_norm/{name}", grad_norm, step)

        if log_histograms:
            writer.add_histogram(f"params/{name}", param.detach().float().cpu(), step)
            if param.grad is not None:
                writer.add_histogram(f"grads/{name}", param.grad.detach().float().cpu(), step)

    safe_add_scalar(writer, "debug/param_norm_total", math.sqrt(total_param_norm_sq), step)
    safe_add_scalar(writer, "debug/grad_norm_total", math.sqrt(total_grad_norm_sq), step)


def binary_auc_score(labels: torch.Tensor, preds: torch.Tensor) -> float:
    """纯 PyTorch 二分类 AUC。

    当某个任务的 label 全为 0 或全为 1 时，AUC 不可定义，返回 NaN。
    """

    labels = labels.detach().float().cpu()
    preds = preds.detach().float().cpu()
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = torch.argsort(preds)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, preds.numel() + 1, dtype=torch.float32)
    pos_rank_sum = ranks[pos].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def grouped_auc_score(labels: torch.Tensor, preds: torch.Tensor, group_ids: Sequence[str]) -> float:
    """按 group 加权平均的 AUC。

    每个 group 内必须同时包含正负样本，否则该 group 的 AUC 不可定义并被跳过。
    最终按有效 group 的样本数加权平均，推荐场景中 GAUC/UAUC 常用这种形式。
    """

    if not group_ids or len(group_ids) != labels.numel():
        return float("nan")

    labels = labels.detach().float().cpu()
    preds = preds.detach().float().cpu()
    group_to_indices: Dict[str, list[int]] = {}
    for idx, group_id in enumerate(group_ids):
        if group_id == "":
            continue
        group_to_indices.setdefault(str(group_id), []).append(idx)

    weighted_auc_sum = 0.0
    weight_sum = 0
    for indices in group_to_indices.values():
        if len(indices) < 2:
            continue
        index_tensor = torch.tensor(indices, dtype=torch.long)
        group_labels = labels[index_tensor]
        group_preds = preds[index_tensor]
        group_auc = binary_auc_score(group_labels, group_preds)
        if math.isnan(group_auc):
            continue
        group_weight = len(indices)
        weighted_auc_sum += group_auc * group_weight
        weight_sum += group_weight

    return weighted_auc_sum / weight_sum if weight_sum > 0 else float("nan")


def compute_metrics(
    labels: torch.Tensor,
    logits: torch.Tensor,
    task_names: Iterable[str],
    trace_ids: Optional[Sequence[str]] = None,
    user_ids: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """计算 loss 之外的评估指标。"""

    probs = torch.sigmoid(logits)
    metrics: Dict[str, float] = {}
    auc_values = []
    gauc_values = []
    uauc_values = []
    for idx, task_name in enumerate(task_names):
        task_auc = binary_auc_score(labels[:, idx], probs[:, idx])
        metrics[f"{task_name}_auc"] = task_auc
        if not math.isnan(task_auc):
            auc_values.append(task_auc)

        task_gauc = grouped_auc_score(labels[:, idx], probs[:, idx], trace_ids or [])
        metrics[f"{task_name}_gauc"] = task_gauc
        if not math.isnan(task_gauc):
            gauc_values.append(task_gauc)

        task_uauc = grouped_auc_score(labels[:, idx], probs[:, idx], user_ids or [])
        metrics[f"{task_name}_uauc"] = task_uauc
        if not math.isnan(task_uauc):
            uauc_values.append(task_uauc)

    metrics["auc"] = float(sum(auc_values) / len(auc_values)) if auc_values else float("nan")
    metrics["gauc"] = float(sum(gauc_values) / len(gauc_values)) if gauc_values else float("nan")
    metrics["uauc"] = float(sum(uauc_values) / len(uauc_values)) if uauc_values else float("nan")
    return metrics


class SigmoidFocalLoss(nn.Module):
    """多目标二分类 Focal Loss。

    Focal Loss 会降低易分类样本的 loss 权重，适合正负极不均衡、长尾目标或
    hard sample 价值更高的推荐场景。输入 logits 和 labels 形状均为 [B, T]。
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        task_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("focal loss reduction 仅支持 mean/sum/none")
        if alpha is not None:
            self.register_buffer("alpha", alpha.view(1, -1))
        else:
            self.alpha = None
        if task_weights is not None:
            self.register_buffer("task_weights", task_weights.view(1, -1))
        else:
            self.task_weights = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(dtype=logits.dtype)
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        p_t = probs * labels + (1.0 - probs) * (1.0 - labels)
        focal_factor = (1.0 - p_t).clamp_min(1e-8).pow(self.gamma)
        loss = focal_factor * bce_loss

        if self.alpha is not None:
            alpha_t = self.alpha * labels + (1.0 - self.alpha) * (1.0 - labels)
            loss = alpha_t * loss
        if self.task_weights is not None:
            loss = loss * self.task_weights

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def build_loss(config: Dict, device: torch.device) -> nn.Module:
    """构造多目标 loss。

    默认使用 BCEWithLogitsLoss；配置 `train.use_focal_loss: true` 后切换为
    SigmoidFocalLoss。两者都支持 [B, num_tasks] 的单/多目标训练。
    """

    train_cfg = config.get("train", {})
    if bool(train_cfg.get("use_focal_loss", False)):
        alpha = train_cfg.get("focal_alpha")
        task_weights = train_cfg.get("task_weights")
        alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device=device) if alpha is not None else None
        task_weight_tensor = (
            torch.tensor(task_weights, dtype=torch.float32, device=device) if task_weights is not None else None
        )
        return SigmoidFocalLoss(
            alpha=alpha_tensor,
            gamma=float(train_cfg.get("focal_gamma", 2.0)),
            task_weights=task_weight_tensor,
            reduction=train_cfg.get("loss_reduction", "mean"),
        )

    pos_weight = train_cfg.get("pos_weight")
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    else:
        pos_weight_tensor = None
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)


def unwrap_model(model: nn.Module) -> nn.Module:
    """torch.compile 包装后，真实模型通常在 _orig_mod 中。"""

    return getattr(model, "_orig_mod", model)


def model_forward(model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
    """兼容不同输入协议的模型前向。

    - DeepFM 使用 libsvm 稀疏输入：feature_ids/feature_values/mask。
    - StructuredInputMMOEModel 直接消费完整 batch，包含 dense、one-hot、
      multi-hot、sequence、pretrained embedding 等结构化字段。
    """

    if getattr(unwrap_model(model), "expects_batch", False):
        return model(batch)
    return model(
        feature_ids=batch["feature_ids"],
        feature_values=batch["feature_values"],
        mask=batch["mask"],
    )


def resolve_device(device_name: str) -> torch.device:
    """解析训练设备。

    生产配置通常会写 cuda；本地或 CI 没有 GPU 时自动回退到 CPU，方便冒烟测试。
    """

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("配置请求使用 CUDA，但当前环境未检测到 GPU，已自动回退到 CPU")
        return torch.device("cpu")
    return torch.device(device_name)


class LossWarmupScheduler:
    """训练 loss warmup 调度器。

    该调度器只缩放训练时用于反传的最终 loss，不改变验证/测试 loss。典型用法是
    多目标或复杂多分支模型训练初期先用较小 loss，降低 early step 的梯度冲击。
    """

    def __init__(
        self,
        steps: int = 0,
        start_factor: float = 0.1,
        end_factor: float = 1.0,
    ) -> None:
        self.steps = max(0, int(steps))
        self.start_factor = float(start_factor)
        self.end_factor = float(end_factor)
        self.step_count = 0
        self.last_factor = 1.0

    @property
    def enabled(self) -> bool:
        return self.steps > 0

    def step(self) -> float:
        if not self.enabled:
            self.last_factor = 1.0
            return 1.0
        self.step_count += 1
        progress = min(self.step_count / self.steps, 1.0)
        self.last_factor = self.start_factor + (self.end_factor - self.start_factor) * progress
        return self.last_factor


def build_loss_warmup_scheduler(train_cfg: Dict) -> Optional[LossWarmupScheduler]:
    """根据训练配置创建 loss warmup 调度器。"""

    warmup_cfg = train_cfg.get("loss_warmup", {})
    enabled = bool(warmup_cfg.get("enabled", False)) or int(train_cfg.get("loss_warmup_steps", 0) or 0) > 0
    if not enabled:
        return None
    return LossWarmupScheduler(
        steps=int(warmup_cfg.get("steps", train_cfg.get("loss_warmup_steps", 0))),
        start_factor=float(warmup_cfg.get("start_factor", train_cfg.get("loss_warmup_start_factor", 0.1))),
        end_factor=float(warmup_cfg.get("end_factor", train_cfg.get("loss_warmup_end_factor", 1.0))),
    )


def run_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    log_interval: int = 100,
    loss_warmup_scheduler: Optional[LossWarmupScheduler] = None,
    writer: Any = None,
    phase: str = "train",
    epoch: int = 0,
    global_step: int = 0,
    tb_log_interval: int = 50,
    tb_log_histograms: bool = False,
) -> Tuple[float, Dict[str, float], int]:
    """训练或评估一个 epoch。optimizer 为空时进入评估模式。"""

    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0
    all_labels = []
    all_logits = []
    all_trace_ids: list[str] = []
    all_user_ids: list[str] = []

    for step, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)
        labels = batch["labels"]
        warmup_factor = 1.0

        with torch.set_grad_enabled(is_train):
            with autocast(device_type=device.type, enabled=use_amp and device.type == "cuda", dtype=amp_dtype):
                logits = model_forward(model, batch)
                loss = criterion(logits, labels)
                if is_train and loss_warmup_scheduler is not None:
                    warmup_factor = loss_warmup_scheduler.step()
                    loss = loss * warmup_factor

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                global_step += 1
                if writer is not None and tb_log_interval > 0 and global_step % tb_log_interval == 0:
                    safe_add_scalar(writer, f"{phase}/loss_step", loss.detach().float().cpu(), global_step)
                    safe_add_scalar(writer, f"{phase}/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                    safe_add_scalar(writer, f"{phase}/loss_warmup_factor", warmup_factor, global_step)
                    safe_add_scalar(writer, f"{phase}/grad_norm_clipped", grad_norm, global_step)
                    writer.add_histogram(f"{phase}/logits", logits.detach().float().cpu(), global_step)
                    writer.add_histogram(f"{phase}/probs", torch.sigmoid(logits.detach().float()).cpu(), global_step)
                    log_model_debug_info(
                        writer,
                        model,
                        global_step,
                        log_histograms=tb_log_histograms,
                    )

        batch_size = labels.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size
        all_labels.append(labels.detach().cpu())
        all_logits.append(logits.detach().float().cpu())
        all_trace_ids.extend(batch.get("trace_ids", [""] * batch_size))
        all_user_ids.extend(batch.get("user_ids", [""] * batch_size))

        if is_train and log_interval > 0 and step % log_interval == 0:
            print(f"step={step} loss={total_loss / max(total_count, 1):.6f}")

    avg_loss = total_loss / max(total_count, 1)
    labels_tensor = torch.cat(all_labels, dim=0)
    logits_tensor = torch.cat(all_logits, dim=0)
    raw_model = unwrap_model(model)
    task_names = getattr(raw_model, "task_names", [f"task_{i}" for i in range(labels_tensor.shape[1])])
    metrics = compute_metrics(
        labels_tensor,
        logits_tensor,
        task_names,
        trace_ids=all_trace_ids,
        user_ids=all_user_ids,
    )
    metrics["loss"] = avg_loss
    return avg_loss, metrics, global_step


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def load_checkpoint(path: str | Path, model: nn.Module, device: torch.device) -> Dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def train(config: Dict) -> None:
    """完整训练流程：读取数据、训练、验证、保存 best/latest、可选测试和导出。"""

    set_seed(int(config.get("seed", 2026)))
    dataloaders, data_info = build_dataloaders(config)

    model_cfg = config.setdefault("model", {})
    model_cfg["num_features"] = int(model_cfg.get("num_features") or data_info.num_features)
    model_cfg["max_features_per_sample"] = int(
        model_cfg.get("max_features_per_sample") or config.get("data", {}).get("max_features_per_sample") or data_info.max_nnz
    )
    if "task_names" not in model_cfg:
        model_cfg["task_names"] = [f"task_{idx}" for idx in range(data_info.label_dim)]
    if len(model_cfg["task_names"]) != data_info.label_dim:
        raise ValueError("model.task_names 数量必须与 label 维度一致")

    train_cfg = config.get("train", {})
    device = resolve_device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.benchmark = bool(train_cfg.get("cudnn_benchmark", True))

    model = build_model(config).to(device)
    if bool(train_cfg.get("torch_compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )
    criterion = build_loss(config, device)
    loss_warmup_scheduler = build_loss_warmup_scheduler(train_cfg)

    # 是否使用tensorboard
    writer = build_summary_writer(config)
    tb_cfg = config.get("debug", {}).get("tensorboard", {})
    tb_log_interval = int(tb_cfg.get("log_interval", train_cfg.get("log_interval", 100)))
    tb_log_histograms = bool(tb_cfg.get("log_histograms", False))
    global_step = 0

    # 是否使用混合精度训练
    use_amp = bool(train_cfg.get("use_amp", False))
    amp_dtype = torch.bfloat16 if train_cfg.get("amp_dtype", "float16") == "bfloat16" else torch.float16
    scaler = GradScaler("cuda", enabled=use_amp and device.type == "cuda" and amp_dtype == torch.float16)

    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    best_metric_name = train_cfg.get("best_metric", "auc")
    best_metric = -float("inf")
    patience = int(train_cfg.get("early_stop_patience", 0))
    bad_epochs = 0

    try:
        for epoch in range(1, int(train_cfg.get("epochs", 3)) + 1):
            train_loss, train_metrics, global_step = run_one_epoch(
                model,
                dataloaders["train"],
                criterion,
                device,
                optimizer=optimizer,
                scaler=scaler,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                log_interval=int(train_cfg.get("log_interval", 100)),
                loss_warmup_scheduler=loss_warmup_scheduler,
                writer=writer,
                phase="train",
                epoch=epoch,
                global_step=global_step,
                tb_log_interval=tb_log_interval,
                tb_log_histograms=tb_log_histograms,
            )
            log_metrics(writer, "train_epoch", train_metrics, epoch)
            print(f"epoch={epoch} train_loss={train_loss:.6f} train_metrics={train_metrics}")

            if "valid" in dataloaders:
                with torch.no_grad():
                    _, valid_metrics, global_step = run_one_epoch(
                        model,
                        dataloaders["valid"],
                        criterion,
                        device,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        writer=writer,
                        phase="valid",
                        epoch=epoch,
                        global_step=global_step,
                    )
                log_metrics(writer, "valid_epoch", valid_metrics, epoch)
                print(f"epoch={epoch} valid_metrics={valid_metrics}")
                current_metric = valid_metrics.get(best_metric_name, -valid_metrics["loss"])
            else:
                valid_metrics = train_metrics
                current_metric = valid_metrics.get(best_metric_name, -valid_metrics["loss"])

            save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, epoch, valid_metrics, config)
            if current_metric > best_metric:
                best_metric = current_metric
                bad_epochs = 0
                save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, valid_metrics, config)
                print(f"保存新的 best checkpoint: metric={best_metric_name}, value={best_metric:.6f}")
            else:
                bad_epochs += 1
                if patience > 0 and bad_epochs >= patience:
                    print(f"触发 early stopping，连续 {bad_epochs} 个 epoch 未提升")
                    break

        best_path = checkpoint_dir / "best.pt"
        if "test" in dataloaders and best_path.exists():
            load_checkpoint(best_path, model, device)
            with torch.no_grad():
                _, test_metrics, global_step = run_one_epoch(
                    model,
                    dataloaders["test"],
                    criterion,
                    device,
                    writer=writer,
                    phase="test",
                    global_step=global_step,
                )
            log_metrics(writer, "test", test_metrics, global_step)
            print(f"test_metrics={test_metrics}")

        if bool(train_cfg.get("export_after_train", True)):
            export_model(config, checkpoint=best_path)
    finally:
        if writer is not None:
            writer.flush()
            writer.close()


def evaluate(config: Dict, split: str, checkpoint: str) -> None:
    """验证/测试入口。"""

    dataloaders, data_info = build_dataloaders(config)
    if split not in dataloaders:
        raise ValueError(f"配置中没有 {split} 数据")
    model_cfg = config.setdefault("model", {})
    model_cfg["num_features"] = int(model_cfg.get("num_features") or data_info.num_features)
    model_cfg["max_features_per_sample"] = int(
        model_cfg.get("max_features_per_sample") or config.get("data", {}).get("max_features_per_sample") or data_info.max_nnz
    )
    model_cfg.setdefault("task_names", [f"task_{idx}" for idx in range(data_info.label_dim)])

    device = resolve_device(config.get("train", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config).to(device)
    load_checkpoint(checkpoint, model, device)
    criterion = build_loss(config, device)
    with torch.no_grad():
        _, metrics, _ = run_one_epoch(model, dataloaders[split], criterion, device)
    print(f"{split}_metrics={metrics}")


def export_model(config: Dict, checkpoint: str | Path) -> None:
    """导出 PyTorch checkpoint 与 TorchScript trace。"""

    export_cfg = config.get("export", {})
    export_dir = Path(export_cfg.get("export_dir", "exports"))
    export_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    # 独立导出时，命令行传入的 yaml 可能没有 num_features；训练保存的 checkpoint
    # 已包含自动推断后的完整配置，因此优先使用 checkpoint 中的 config。
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    export_config = checkpoint_data.get("config", config)
    model = build_model(export_config).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": export_config,
        },
        export_dir / "model_state.pt",
    )

    if not getattr(model, "expects_batch", False):
        max_features = int(export_config["model"].get("max_features_per_sample", 64))
        example_feature_ids = torch.ones(1, max_features, dtype=torch.long)
        example_feature_values = torch.ones(1, max_features, dtype=torch.float32)
        example_mask = torch.ones(1, max_features, dtype=torch.bool)

        traced = torch.jit.trace(model, (example_feature_ids, example_feature_values, example_mask))
        traced.save(str(export_dir / "deepfm_traced.pt"))
    else:
        print("结构化 batch 模型已导出 state_dict；TorchScript trace 已跳过。")

    with (export_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(export_config, f, ensure_ascii=False, indent=2)
    print(f"模型已导出到: {export_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch DeepFM 训练入口")
    parser.add_argument("--common-config", type=str, default="conf/common.yaml", help="通用配置文件路径")
    parser.add_argument("--model-config", type=str, default="conf/model.yaml", help="模型配置文件路径")
    parser.add_argument("--config", type=str, default="", help="可选覆盖配置文件路径，兼容旧版完整配置")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "valid", "test", "export"])
    parser.add_argument("--checkpoint", type=str, default="", help="评估或导出时使用的 checkpoint")
    return parser.parse_args()


def main() -> None:
    validate_runtime()
    args = parse_args()
    config = load_runtime_config(args.common_config, args.model_config, args.config or None)
    if args.mode == "train":
        train(config)
    elif args.mode in {"valid", "test"}:
        checkpoint = args.checkpoint or str(Path(config.get("train", {}).get("checkpoint_dir", "checkpoints")) / "best.pt")
        evaluate(config, args.mode, checkpoint)
    elif args.mode == "export":
        checkpoint = args.checkpoint or str(Path(config.get("train", {}).get("checkpoint_dir", "checkpoints")) / "best.pt")
        export_model(config, checkpoint)


if __name__ == "__main__":
    main()
