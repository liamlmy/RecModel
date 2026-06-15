"""DeepFM 模型定义。

模型输入来自 core.data 的 batch：
- feature_ids: [B, F]，padding ID 为 0。
- feature_values: [B, F]，libsvm 中的特征值。
- mask: [B, F]，有效特征位置。

DeepFM 输出 logits，训练阶段建议使用 BCEWithLogitsLoss。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

from core.structure import LHUC, MMOE, MLP, MultiHeadSelfAttention, PEPNetGate, RankMixer, SENetLayer


class SparseInputLayer(nn.Module):
    """稀疏输入层：同时提供一阶权重和高阶 embedding。"""

    def __init__(self, num_features: int, embedding_dim: int, num_tasks: int) -> None:
        super().__init__()
        self.linear_embedding = nn.Embedding(num_features, num_tasks, padding_idx=0)
        self.embedding = nn.Embedding(num_features, embedding_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.linear_embedding.weight)

    def forward(
        self,
        feature_ids: torch.Tensor,
        feature_values: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        values = feature_values.unsqueeze(-1)
        linear_terms = self.linear_embedding(feature_ids) * values
        embeddings = self.embedding(feature_ids) * values
        return {
            "linear_terms": linear_terms,
            "embeddings": embeddings,
        }


class FMInteraction(nn.Module):
    """FM 二阶交叉项。

    标准公式：0.5 * ((sum x)^2 - sum(x^2))。这里返回 [B, D] 的交叉向量，
    再由任务输出层映射到单/多目标 logits。
    """

    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            embeddings = embeddings.masked_fill(~mask.unsqueeze(-1), 0.0)
        square_of_sum = torch.sum(embeddings, dim=1).pow(2)
        sum_of_square = torch.sum(embeddings.pow(2), dim=1)
        return 0.5 * (square_of_sum - sum_of_square)


class DeepFM(nn.Module):
    """支持单目标/多目标、MMOE、SENet、LHUC、PEPNet、RankMixer 的 DeepFM。"""

    def __init__(self, config: Dict) -> None:
        super().__init__()
        model_cfg = config.get("model", {})
        self.num_features = int(model_cfg["num_features"])
        self.embedding_dim = int(model_cfg.get("embedding_dim", 16))
        self.task_names: List[str] = list(model_cfg.get("task_names", ["label"]))
        self.num_tasks = len(self.task_names)
        self.dnn_input_mode = model_cfg.get("dnn_input_mode", "pool")
        self.max_features_per_sample = int(model_cfg.get("max_features_per_sample", 64))

        self.input_layer = SparseInputLayer(self.num_features, self.embedding_dim, self.num_tasks)
        self.fm = FMInteraction()
        self.fm_output = nn.Linear(self.embedding_dim, self.num_tasks, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.num_tasks))

        self.use_senet = bool(model_cfg.get("use_senet", False))
        self.use_rankmixer = bool(model_cfg.get("use_rankmixer", False))
        self.use_self_attention = bool(model_cfg.get("use_self_attention", False))
        self.use_lhuc = bool(model_cfg.get("use_lhuc", False))
        self.use_pepnet = bool(model_cfg.get("use_pepnet", False))
        self.use_mmoe = bool(model_cfg.get("use_mmoe", self.num_tasks > 1))
        self.stop_gradient = set(model_cfg.get("stop_gradient", []))

        if self.use_senet:
            self.senet = SENetLayer(field_size=self.max_features_per_sample)
        else:
            self.senet = None

        if self.use_rankmixer:
            self.rankmixer = RankMixer(
                field_size=self.max_features_per_sample,
                embed_dim=self.embedding_dim,
                num_layers=int(model_cfg.get("rankmixer_layers", 2)),
                token_mlp_dim=int(model_cfg.get("rankmixer_token_dim", 64)),
                channel_mlp_dim=int(model_cfg.get("rankmixer_channel_dim", 128)),
                dropout=float(model_cfg.get("dropout", 0.0)),
            )
        else:
            self.rankmixer = None

        if self.use_self_attention:
            self.self_attention = MultiHeadSelfAttention(
                embed_dim=self.embedding_dim,
                num_heads=int(model_cfg.get("attention_num_heads", 4)),
                dropout=float(model_cfg.get("dropout", 0.0)),
                use_flash_attention=bool(model_cfg.get("use_flash_attention", False)),
            )
        else:
            self.self_attention = None

        if self.dnn_input_mode == "flatten":
            dnn_input_dim = self.max_features_per_sample * self.embedding_dim
        elif self.dnn_input_mode == "pool":
            # sum pooling、mean pooling、FM interaction vector 三路拼接。
            dnn_input_dim = self.embedding_dim * 3
        else:
            raise ValueError("model.dnn_input_mode 仅支持 pool 或 flatten")

        if self.use_pepnet:
            self.pepnet_gate = PEPNetGate(dnn_input_dim, dnn_input_dim)
        else:
            self.pepnet_gate = None

        if self.use_lhuc:
            self.lhuc = LHUC(dnn_input_dim, dnn_input_dim)
        else:
            self.lhuc = None

        hidden_units: Sequence[int] = model_cfg.get("dnn_hidden_units", [256, 128, 64])
        dropout = float(model_cfg.get("dropout", 0.1))
        activation = model_cfg.get("activation", "relu")
        use_batch_norm = bool(model_cfg.get("use_batch_norm", False))
        norm_type = model_cfg.get("norm_type", "batchnorm" if use_batch_norm else "none")

        if self.use_mmoe:
            expert_output_dim = int(model_cfg.get("mmoe_expert_output_dim", 64))
            self.mmoe = MMOE(
                input_dim=dnn_input_dim,
                expert_hidden_units=model_cfg.get("mmoe_expert_hidden_units", [128, 64]),
                expert_output_dim=expert_output_dim,
                num_experts=int(model_cfg.get("mmoe_num_experts", 4)),
                num_tasks=self.num_tasks,
                dropout=dropout,
                norm_type=norm_type,
            )
            self.task_towers = nn.ModuleList(
                [
                    MLP(
                        expert_output_dim,
                        model_cfg.get("tower_hidden_units", [64, 32]),
                        output_dim=1,
                        activation=activation,
                        dropout=dropout,
                        use_batch_norm=use_batch_norm,
                        norm_type=norm_type,
                    )
                    for _ in range(self.num_tasks)
                ]
            )
            self.deep = None
        else:
            self.deep = MLP(
                dnn_input_dim,
                hidden_units,
                output_dim=self.num_tasks,
                activation=activation,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                norm_type=norm_type,
            )
            self.mmoe = None
            self.task_towers = None

    def _maybe_stop_gradient(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """按配置截断某个结构输出的梯度。

        例如 `stop_gradient: ["fm_vector", "senet"]` 会让后续 loss 不再反传到
        对应结构之前的计算图。这个开关适合做消融、冻结预训练结构输出，或稳定
        多分支联合训练。
        """

        return tensor.detach() if name in self.stop_gradient else tensor

    def _fit_fixed_field_size(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """把 batch 内动态 field 数对齐到配置中的 max_features_per_sample。

        SENet、RankMixer、flatten DNN 都需要固定 field_size。collate_fn 已经会
        尽量截断，这里再兜底做 pad/truncate，保证模型可独立使用。
        """

        batch_size, field_size, embed_dim = embeddings.shape
        target_size = self.max_features_per_sample
        if field_size == target_size:
            return embeddings, mask
        if field_size > target_size:
            return embeddings[:, :target_size, :], mask[:, :target_size]

        pad_len = target_size - field_size
        pad_embeddings = embeddings.new_zeros(batch_size, pad_len, embed_dim)
        pad_mask = mask.new_zeros(batch_size, pad_len)
        return torch.cat([embeddings, pad_embeddings], dim=1), torch.cat([mask, pad_mask], dim=1)

    def _build_dnn_input(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        fm_vector: torch.Tensor,
    ) -> torch.Tensor:
        if self.dnn_input_mode == "flatten":
            fixed_embeddings, _ = self._fit_fixed_field_size(embeddings, mask)
            return fixed_embeddings.flatten(start_dim=1)

        masked_embeddings = embeddings.masked_fill(~mask.unsqueeze(-1), 0.0)
        sum_pooling = torch.sum(masked_embeddings, dim=1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(masked_embeddings.dtype)
        mean_pooling = sum_pooling / denom
        return torch.cat([sum_pooling, mean_pooling, fm_vector], dim=-1)

    def forward(
        self,
        feature_ids: torch.Tensor,
        feature_values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = feature_ids.ne(0)

        sparse_outputs = self.input_layer(feature_ids, feature_values)
        linear_terms = sparse_outputs["linear_terms"].masked_fill(~mask.unsqueeze(-1), 0.0)
        embeddings = sparse_outputs["embeddings"].masked_fill(~mask.unsqueeze(-1), 0.0)
        linear_terms = self._maybe_stop_gradient("linear_terms", linear_terms)
        embeddings = self._maybe_stop_gradient("embeddings", embeddings)

        linear_logit = torch.sum(linear_terms, dim=1)
        linear_logit = self._maybe_stop_gradient("linear_logit", linear_logit)

        fixed_embeddings, fixed_mask = self._fit_fixed_field_size(embeddings, mask)
        if self.senet is not None:
            fixed_embeddings = self.senet(fixed_embeddings, fixed_mask)
            fixed_embeddings = self._maybe_stop_gradient("senet", fixed_embeddings)
        if self.rankmixer is not None:
            fixed_embeddings = self.rankmixer(fixed_embeddings, fixed_mask)
            fixed_embeddings = self._maybe_stop_gradient("rankmixer", fixed_embeddings)
        if self.self_attention is not None:
            fixed_embeddings = self.self_attention(fixed_embeddings, fixed_mask)
            fixed_embeddings = self._maybe_stop_gradient("self_attention", fixed_embeddings)
        fixed_embeddings = self._maybe_stop_gradient("feature_interaction", fixed_embeddings)

        fm_vector = self.fm(fixed_embeddings, fixed_mask)
        fm_vector = self._maybe_stop_gradient("fm_vector", fm_vector)
        fm_logit = self.fm_output(fm_vector)
        fm_logit = self._maybe_stop_gradient("fm_logit", fm_logit)

        dnn_input = self._build_dnn_input(fixed_embeddings, fixed_mask, fm_vector)
        dnn_input = self._maybe_stop_gradient("dnn_input", dnn_input)
        if self.pepnet_gate is not None:
            dnn_input = self.pepnet_gate(dnn_input, dnn_input)
            dnn_input = self._maybe_stop_gradient("pepnet", dnn_input)
        if self.lhuc is not None:
            dnn_input = self.lhuc(dnn_input, dnn_input)
            dnn_input = self._maybe_stop_gradient("lhuc", dnn_input)

        if self.mmoe is not None and self.task_towers is not None:
            task_representations = self.mmoe(dnn_input)
            if "mmoe" in self.stop_gradient:
                task_representations = [task_repr.detach() for task_repr in task_representations]
            deep_logit = torch.cat(
                [tower(task_repr) for tower, task_repr in zip(self.task_towers, task_representations)],
                dim=-1,
            )
        else:
            assert self.deep is not None
            deep_logit = self.deep(dnn_input)
        deep_logit = self._maybe_stop_gradient("deep_logit", deep_logit)

        return linear_logit + fm_logit + deep_logit + self.bias


def build_model(config: Dict) -> DeepFM:
    """模型构造入口，方便 core.main 或线上服务统一调用。"""

    return DeepFM(config)
