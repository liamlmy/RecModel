"""libsvm 数据读取、采样、特征/标签识别工具。

本文件刻意不依赖 pandas，原因是生产推荐样本通常很大，逐行解析更容易做
流式扩展和内存控制。当前实现是 map-style Dataset：会在初始化时把样本解析
到内存中，适合中小规模实验；如果线上样本特别大，可以沿用 parse_libsvm_line
和 collate_fn 改成 IterableDataset。
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class DataInfo:
    """训练数据推断出的元信息。

    Attributes:
        num_features: embedding 表大小。由于 0 被保留为 padding，下游会把原始
            feature_id 整体 +1，所以 num_features = max_feature_id + 2。
        label_dim: 标签维度，单目标为 1，多目标为任务数。
        max_nnz: 单条样本中出现过的最大非零特征数。
        num_samples: 当前数据集样本数。
    """

    num_features: int
    label_dim: int
    max_nnz: int
    num_samples: int


@dataclass
class LibSVMSample:
    """单条 libsvm 样本。"""

    labels: List[float]
    feature_ids: List[int]
    feature_values: List[float]
    trace_id: str = ""
    user_id: str = ""


def prepare_input_path(path: str | Path, config: Mapping[str, Any]) -> Path:
    """根据配置把输入路径准备成本地可读路径。

    - local: 直接返回原始本地路径。
    - hdfs: 使用 `hdfs dfs -get -f` 拉取到本地 cache 后返回缓存路径。

    训练代码仍然只处理本地文件句柄，避免把 HDFS 读取细节散落到 Dataset 中。
    """

    path_text = str(path)
    data_cfg = config.get("data", {})
    storage_cfg = config.get("storage", {})
    path_type = data_cfg.get("path_type", storage_cfg.get("input_type", "local"))
    path_type = str(path_type).lower()
    if path_type == "local":
        return Path(path_text)
    if path_type != "hdfs":
        raise ValueError(f"不支持的输入路径类型: {path_type}")

    cache_dir = Path(storage_cfg.get("local_cache_dir", ".cache/hdfs"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / Path(path_text.rstrip("/")).name
    hdfs_cmd = storage_cfg.get("hdfs_cmd", "hdfs")
    command = [hdfs_cmd, "dfs", "-get", "-f", path_text, str(local_path)]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"未找到 HDFS 命令 `{hdfs_cmd}`，请检查 storage.hdfs_cmd") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"HDFS 拉取失败: {' '.join(command)}") from exc
    return local_path


@dataclass
class FeatureSpec:
    """业务特征处理配置。

    推荐样本在进入模型前通常会被转成 `feature_id + feature_value` 的稀疏
    token。这个配置用于描述每个原始字段如何转成稀疏 token。

    Attributes:
        name: 原始特征名，例如 `user_age`、`city_id`、`tag_ids`。
        feature_type: 特征类型，支持 numeric/one_hot/multi_hot/sequence/pretrained_embedding。
        feature_id: 数值特征的固定特征 ID。数值特征一般用同一个 ID 承载不同
            value，例如 `age_feature_id: normalized_age`。
        vocab: 类别值到特征 ID 的映射。适合离线构建词表的 one-hot/multi-hot。
        hash_bucket_size: 没有词表时使用稳定 hash 分桶。
        hash_bucket_offset: hash 分桶的 ID 起点。注意 0 被保留为 padding。
        default_id: OOV 或缺失值对应 ID；为空时会跳过缺失/OOV。
        max_len: multi-hot/sequence 最大保留长度。
        delimiter: 字符串 multi-hot/sequence 的分隔符。
        value_delimiter: `id:value` 形式的分隔符。
        side_info: 序列特征中每个 item 的 side info 配置，例如 cate/brand。
        embedding_dim: 预训练 embedding 的维度。
        embedding_delimiter: 字符串 embedding 向量的分隔符。
        fill_missing: 预训练 embedding 缺失时是否补零。
        l2_normalize: 是否对预训练 embedding 做 L2 归一化。
    """

    name: str
    feature_type: str
    feature_id: Optional[int] = None
    vocab: Optional[Mapping[str, int]] = None
    hash_bucket_size: int = 0
    hash_bucket_offset: int = 1
    default_id: Optional[int] = None
    max_len: int = 0
    delimiter: str = ","
    value_delimiter: str = ":"
    side_info: Optional[Mapping[str, "FeatureSpec"]] = None
    embedding_dim: int = 0
    embedding_delimiter: str = ","
    fill_missing: bool = True
    l2_normalize: bool = False


@dataclass
class SparseFeatureBundle:
    """DeepFM 可直接消费的稀疏 token 集合。"""

    feature_ids: List[int]
    feature_values: List[float]


@dataclass
class SequenceFeatureBundle:
    """序列特征处理结果。

    item_feature_ids/item_feature_values 表示序列主 ID；side_feature_ids 是
    `side_name -> [T]`，用于 DIN/DIEN/Transformer 类序列建模模块进一步消费。
    """

    item_feature_ids: List[int]
    item_feature_values: List[float]
    side_feature_ids: Dict[str, List[int]]
    side_feature_values: Dict[str, List[float]]


@dataclass
class ProcessedFeatureSample:
    """结构化原始样本处理后的结果。"""

    labels: List[float]
    sparse: SparseFeatureBundle
    sequences: Dict[str, SequenceFeatureBundle]
    pretrained_embeddings: Dict[str, List[float]]


def stable_hash(text: Any, bucket_size: int) -> int:
    """稳定 hash，避免 Python 内置 hash 因进程随机种子导致线上线下不一致。"""

    if bucket_size <= 0:
        raise ValueError("bucket_size 必须大于 0")
    digest = hashlib.md5(str(text).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % bucket_size


def _to_list(value: Any, delimiter: str = ",") -> List[Any]:
    """把字符串、tuple、set、list 等输入统一转成 list。"""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple) or isinstance(value, set):
        return list(value)
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return []
        return [item for item in value.split(delimiter) if item != ""]
    return [value]


def _parse_id_value(token: Any, value_delimiter: str = ":") -> Tuple[Any, float]:
    """解析 `id` 或 `id:value` token。

    multi-hot 和序列特征常会带权重，例如兴趣 tag 权重、行为衰减权重。
    """

    if isinstance(token, Mapping):
        raw_id = token.get("id", token.get("value"))
        weight = float(token.get("weight", token.get("score", 1.0)))
        return raw_id, weight

    if isinstance(token, str) and value_delimiter in token:
        raw_id, raw_value = token.split(value_delimiter, 1)
        return raw_id, float(raw_value)

    return token, 1.0


def lookup_feature_id(raw_value: Any, spec: FeatureSpec) -> Optional[int]:
    """根据词表、hash 或原始整数值获取模型侧 feature_id。

    返回 None 表示该值应被跳过，例如 OOV 且未配置 default_id。
    """

    if raw_value is None or raw_value == "":
        return spec.default_id

    text_value = str(raw_value)
    if spec.vocab is not None:
        return spec.vocab.get(text_value, spec.default_id)

    if spec.hash_bucket_size > 0:
        return spec.hash_bucket_offset + stable_hash(text_value, spec.hash_bucket_size)

    try:
        feature_id = int(raw_value)
    except (TypeError, ValueError):
        return spec.default_id
    return feature_id if feature_id > 0 else spec.default_id


def normalize_numeric_value(
    value: Any,
    mean: float = 0.0,
    std: float = 1.0,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    log1p: bool = False,
) -> float:
    """数值特征归一化。

    Args:
        value: 原始数值。
        mean/std: z-score 参数。std 为 0 时会自动按 1 处理。
        min_value/max_value: 可选截断边界，避免异常值打爆 embedding value。
        log1p: 是否对非负值做 log1p，常用于曝光、点击、价格等长尾数值。
    """

    numeric = float(value)
    if min_value is not None:
        numeric = max(numeric, min_value)
    if max_value is not None:
        numeric = min(numeric, max_value)
    if log1p:
        numeric = torch.log1p(torch.tensor(max(numeric, 0.0))).item()
    safe_std = std if abs(std) > 1e-12 else 1.0
    return (numeric - mean) / safe_std


def process_numeric_feature(
    value: Any,
    spec: FeatureSpec,
    mean: float = 0.0,
    std: float = 1.0,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    log1p: bool = False,
) -> SparseFeatureBundle:
    """处理数值类特征。

    数值类特征输出一个固定 feature_id，feature_value 为归一化后的数值。
    """

    if value is None or value == "":
        return SparseFeatureBundle([], [])
    if spec.feature_id is None:
        raise ValueError(f"数值特征 {spec.name} 必须配置 feature_id")
    normalized = normalize_numeric_value(value, mean, std, min_value, max_value, log1p)
    if normalized == 0.0:
        return SparseFeatureBundle([], [])
    return SparseFeatureBundle([spec.feature_id], [float(normalized)])


def process_one_hot_feature(value: Any, spec: FeatureSpec) -> SparseFeatureBundle:
    """处理 one-hot 类特征。

    one-hot 特征输出一个类别 ID，value 固定为 1。
    """

    feature_id = lookup_feature_id(value, spec)
    if feature_id is None:
        return SparseFeatureBundle([], [])
    return SparseFeatureBundle([feature_id], [1.0])


def process_multi_hot_feature(
    value: Any,
    spec: FeatureSpec,
    normalize: str = "mean",
) -> SparseFeatureBundle:
    """处理 multi-hot 类特征。

    支持输入：
    - `"1,2,3"`
    - `"1:0.2,2:0.8"`
    - `[1, 2, 3]`
    - `[{"id": "sports", "weight": 0.7}]`

    normalize:
        none: 保留原始权重。
        mean: 每个 token 权重除以有效 token 数，适合兴趣 tag 等集合特征。
        sum: 按权重和归一化，适合概率/分布特征。
    """

    tokens = _to_list(value, spec.delimiter)
    if spec.max_len > 0:
        tokens = tokens[: spec.max_len]

    feature_ids: List[int] = []
    feature_values: List[float] = []
    for token in tokens:
        raw_id, weight = _parse_id_value(token, spec.value_delimiter)
        feature_id = lookup_feature_id(raw_id, spec)
        if feature_id is None or weight == 0.0:
            continue
        feature_ids.append(feature_id)
        feature_values.append(float(weight))

    if not feature_ids:
        return SparseFeatureBundle([], [])

    if normalize == "mean":
        denom = float(len(feature_values))
        feature_values = [value / denom for value in feature_values]
    elif normalize == "sum":
        denom = sum(abs(value) for value in feature_values) or 1.0
        feature_values = [value / denom for value in feature_values]
    elif normalize != "none":
        raise ValueError("normalize 仅支持 none/mean/sum")

    return SparseFeatureBundle(feature_ids, feature_values)


def _parse_sequence_item(item: Any, side_delimiter: str = "|") -> Tuple[Any, Dict[str, Any], float]:
    """解析序列 item。

    支持以下格式：
    - `"item_id"`
    - `"item_id:0.8"`，其中 0.8 是行为权重。
    - `"item_id|cate=10|brand=20"`
    - `{"id": "item_id", "weight": 0.8, "cate": 10, "brand": 20}`
    """

    if isinstance(item, Mapping):
        item_id = item.get("id", item.get("item_id"))
        weight = float(item.get("weight", item.get("score", 1.0)))
        side_info = {key: value for key, value in item.items() if key not in {"id", "item_id", "weight", "score"}}
        return item_id, side_info, weight

    if isinstance(item, str) and side_delimiter in item:
        parts = item.split(side_delimiter)
        item_id, weight = _parse_id_value(parts[0])
        side_info: Dict[str, Any] = {}
        for part in parts[1:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            side_info[key] = value
        return item_id, side_info, weight

    item_id, weight = _parse_id_value(item)
    return item_id, {}, weight


def process_sequence_feature(
    value: Any,
    spec: FeatureSpec,
    side_delimiter: str = "|",
) -> SequenceFeatureBundle:
    """处理序列特征，包括序列主 ID 及每个 item 对应的 side info。

    序列特征通常不会直接塞进 DeepFM 的无序稀疏输入，而是给 DIN attention、
    self-attention 或用户行为塔消费。因此这里保留顺序结构。
    """

    items = _to_list(value, spec.delimiter)
    if spec.max_len > 0:
        # 推荐系统里一般保留最近 max_len 个行为。若你的样本是从近到远排列，
        # 可以在上游反转，或者把这里改成 items[:max_len]。
        items = items[-spec.max_len :]

    item_feature_ids: List[int] = []
    item_feature_values: List[float] = []
    side_feature_ids: Dict[str, List[int]] = {name: [] for name in (spec.side_info or {})}
    side_feature_values: Dict[str, List[float]] = {name: [] for name in (spec.side_info or {})}

    for item in items:
        item_id, side_info, weight = _parse_sequence_item(item, side_delimiter=side_delimiter)
        feature_id = lookup_feature_id(item_id, spec)
        if feature_id is None:
            continue
        item_feature_ids.append(feature_id)
        item_feature_values.append(float(weight))

        for side_name, side_spec in (spec.side_info or {}).items():
            side_id = lookup_feature_id(side_info.get(side_name), side_spec)
            side_feature_ids[side_name].append(side_id or 0)
            side_feature_values[side_name].append(1.0 if side_id else 0.0)

    return SequenceFeatureBundle(
        item_feature_ids=item_feature_ids,
        item_feature_values=item_feature_values,
        side_feature_ids=side_feature_ids,
        side_feature_values=side_feature_values,
    )


def parse_embedding_vector(value: Any, delimiter: str = ",") -> List[float]:
    """解析预训练 embedding 向量。

    支持如下输入：
    - `[0.1, 0.2, 0.3]`
    - `"0.1,0.2,0.3"`
    - `"0.1 0.2 0.3"`
    - `{"embedding": [0.1, 0.2, 0.3]}`
    """

    if value is None:
        return []
    if isinstance(value, Mapping):
        value = value.get("embedding", value.get("vector", value.get("value")))
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().cpu().flatten().tolist()]
    if isinstance(value, list) or isinstance(value, tuple):
        return [float(item) for item in value]
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return []
        if delimiter in value:
            return [float(item) for item in value.split(delimiter) if item != ""]
        return [float(item) for item in value.split() if item != ""]
    raise ValueError(f"无法解析 embedding 向量: {value}")


def process_pretrained_embedding_feature(value: Any, spec: FeatureSpec) -> List[float]:
    """处理预训练好的纯 embedding 特征。

    这类特征不进入 ID embedding 表，而是直接作为 dense vector 输入模型后续
    的 dense tower 或与 DNN 输入拼接。缺失时默认按 embedding_dim 补零。
    """

    vector = parse_embedding_vector(value, delimiter=spec.embedding_delimiter)
    if not vector:
        if not spec.fill_missing:
            return []
        if spec.embedding_dim <= 0:
            raise ValueError(f"预训练 embedding 特征 {spec.name} 缺失时必须配置 embedding_dim")
        vector = [0.0] * spec.embedding_dim

    if spec.embedding_dim > 0:
        if len(vector) > spec.embedding_dim:
            vector = vector[: spec.embedding_dim]
        elif len(vector) < spec.embedding_dim:
            vector = vector + [0.0] * (spec.embedding_dim - len(vector))

    if spec.l2_normalize:
        tensor = torch.tensor(vector, dtype=torch.float32)
        norm = torch.linalg.vector_norm(tensor).item()
        if norm > 1e-12:
            vector = [item / norm for item in vector]

    return [float(item) for item in vector]


def process_raw_feature_sample(
    raw_sample: Mapping[str, Any],
    feature_specs: Sequence[FeatureSpec],
    label_names: Sequence[str] = ("label",),
    multi_hot_normalize: str = "mean",
) -> ProcessedFeatureSample:
    """把结构化原始样本处理成稀疏特征和序列特征。

    这个函数适合 JSON/CSV 解析后的 dict 样本。当前 libsvm 训练链路仍使用
    parse_libsvm_line；如果你后续把样本格式升级为结构化字段，可以直接复用
    这里的处理逻辑。
    """

    labels = [float(raw_sample[name]) for name in label_names if name in raw_sample]
    sparse_ids: List[int] = []
    sparse_values: List[float] = []
    sequences: Dict[str, SequenceFeatureBundle] = {}
    pretrained_embeddings: Dict[str, List[float]] = {}

    for spec in feature_specs:
        raw_value = raw_sample.get(spec.name)
        feature_type = spec.feature_type.lower()
        if feature_type == "numeric":
            bundle = process_numeric_feature(raw_value, spec)
        elif feature_type == "one_hot":
            bundle = process_one_hot_feature(raw_value, spec)
        elif feature_type == "multi_hot":
            bundle = process_multi_hot_feature(raw_value, spec, normalize=multi_hot_normalize)
        elif feature_type == "sequence":
            sequences[spec.name] = process_sequence_feature(raw_value, spec)
            continue
        elif feature_type == "pretrained_embedding":
            pretrained_embeddings[spec.name] = process_pretrained_embedding_feature(raw_value, spec)
            continue
        else:
            raise ValueError(f"未知特征类型: {spec.feature_type}")

        sparse_ids.extend(bundle.feature_ids)
        sparse_values.extend(bundle.feature_values)

    return ProcessedFeatureSample(
        labels=labels,
        sparse=SparseFeatureBundle(sparse_ids, sparse_values),
        sequences=sequences,
        pretrained_embeddings=pretrained_embeddings,
    )


def read_jsonl_samples(path: str | Path) -> List[Dict[str, Any]]:
    """读取 JSONL 原始样本文件。

    JSONL 更适合承载序列、side info、预训练 embedding 这类结构化特征；libsvm
    更适合已经离线展开好的稀疏 ID:value。
    """

    samples: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def build_feature_specs(spec_configs: Sequence[Mapping[str, Any]]) -> List[FeatureSpec]:
    """从 dict 配置构造 FeatureSpec 列表。

    JSON/YAML 中的 side_info 也是 dict，需要递归转换成 FeatureSpec，方便直接
    从 schema 文件恢复结构化特征处理规则。
    """

    feature_specs: List[FeatureSpec] = []
    for config in spec_configs:
        config_dict = dict(config)
        side_info = config_dict.get("side_info")
        if side_info is not None:
            config_dict["side_info"] = {
                name: build_feature_specs([side_config])[0]
                for name, side_config in side_info.items()
            }
        feature_specs.append(FeatureSpec(**config_dict))
    return feature_specs


class RawFeatureDataset(Dataset):
    """结构化 JSONL 样本数据集。

    该数据集用于包含数值、one-hot、multi-hot、序列和预训练 embedding 的原始
    样本。现有 DeepFM libsvm 训练链路不受影响；更复杂模型可直接使用本类。
    """

    def __init__(
        self,
        path: str | Path,
        feature_specs: Sequence[FeatureSpec],
        label_names: Sequence[str] = ("label",),
        multi_hot_normalize: str = "mean",
        sample_rate: float = 1.0,
        max_samples: Optional[int] = None,
        seed: int = 2026,
    ) -> None:
        self.path = Path(path)
        self.feature_specs = list(feature_specs)
        self.label_names = list(label_names)
        self.samples: List[ProcessedFeatureSample] = []

        if not self.path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.path}")

        rng = random.Random(seed)
        for raw_sample in read_jsonl_samples(self.path):
            if sample_rate < 1.0 and rng.random() > sample_rate:
                continue
            self.samples.append(
                process_raw_feature_sample(
                    raw_sample,
                    self.feature_specs,
                    label_names=self.label_names,
                    multi_hot_normalize=multi_hot_normalize,
                )
            )
            if max_samples is not None and len(self.samples) >= max_samples:
                break

        if not self.samples:
            raise ValueError(f"没有从 {self.path} 读取到有效样本")

        label_dim = len(self.samples[0].labels)
        for sample in self.samples:
            if len(sample.labels) != label_dim:
                raise ValueError("同一个 JSONL 数据文件内 label 维度不一致")

        self.info = DataInfo(
            num_features=max(
                (max(sample.sparse.feature_ids) if sample.sparse.feature_ids else 0)
                for sample in self.samples
            )
            + 1,
            label_dim=label_dim,
            max_nnz=max(len(sample.sparse.feature_ids) for sample in self.samples),
            num_samples=len(self.samples),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ProcessedFeatureSample:
        return self.samples[index]


def sequence_collate(
    sequences: Sequence[SequenceFeatureBundle],
    max_len: int = 0,
) -> Dict[str, Any]:
    """把 SequenceFeatureBundle padding 成 batch tensor。

    Returns:
        item_feature_ids: [B, T]
        item_feature_values: [B, T]
        item_mask: [B, T]
        side_feature_ids: dict[str, Tensor[B, T]]
        side_feature_values: dict[str, Tensor[B, T]]
    """

    if not sequences:
        raise ValueError("sequences 不能为空")

    batch_size = len(sequences)
    batch_max_len = max(len(seq.item_feature_ids) for seq in sequences)
    if max_len > 0:
        batch_max_len = min(batch_max_len, max_len)
    batch_max_len = max(batch_max_len, 1)

    item_feature_ids = torch.zeros(batch_size, batch_max_len, dtype=torch.long)
    item_feature_values = torch.zeros(batch_size, batch_max_len, dtype=torch.float32)
    item_mask = torch.zeros(batch_size, batch_max_len, dtype=torch.bool)

    side_names = sorted({name for seq in sequences for name in seq.side_feature_ids})
    side_feature_ids = {name: torch.zeros(batch_size, batch_max_len, dtype=torch.long) for name in side_names}
    side_feature_values = {name: torch.zeros(batch_size, batch_max_len, dtype=torch.float32) for name in side_names}

    for row, seq in enumerate(sequences):
        length = min(len(seq.item_feature_ids), batch_max_len)
        if length == 0:
            continue
        item_feature_ids[row, :length] = torch.tensor(seq.item_feature_ids[-length:], dtype=torch.long)
        item_feature_values[row, :length] = torch.tensor(seq.item_feature_values[-length:], dtype=torch.float32)
        item_mask[row, :length] = True
        for side_name in side_names:
            ids = seq.side_feature_ids.get(side_name, [])[-length:]
            values = seq.side_feature_values.get(side_name, [])[-length:]
            if ids:
                side_feature_ids[side_name][row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            if values:
                side_feature_values[side_name][row, : len(values)] = torch.tensor(values, dtype=torch.float32)

    return {
        "item_feature_ids": item_feature_ids,
        "item_feature_values": item_feature_values,
        "item_mask": item_mask,
        "side_feature_ids": side_feature_ids,
        "side_feature_values": side_feature_values,
    }


def pretrained_embedding_collate(
    embeddings: Sequence[Mapping[str, Sequence[float]]],
    embedding_dims: Optional[Mapping[str, int]] = None,
) -> Dict[str, torch.Tensor]:
    """把预训练 embedding 特征拼成 batch tensor。

    Args:
        embeddings: 每条样本的 `feature_name -> vector`。
        embedding_dims: 可选维度声明。声明后缺失样本会补零，并保证输出维度稳定。
    """

    if not embeddings:
        raise ValueError("embeddings 不能为空")

    names = sorted({name for item in embeddings for name in item})
    if embedding_dims is not None:
        names = sorted(set(names) | set(embedding_dims))

    output: Dict[str, torch.Tensor] = {}
    for name in names:
        dim = int(embedding_dims[name]) if embedding_dims and name in embedding_dims else 0
        if dim <= 0:
            dim = max((len(item.get(name, [])) for item in embeddings), default=0)
        dim = max(dim, 1)

        tensor = torch.zeros(len(embeddings), dim, dtype=torch.float32)
        for row, item in enumerate(embeddings):
            vector = list(item.get(name, []))[:dim]
            if vector:
                tensor[row, : len(vector)] = torch.tensor(vector, dtype=torch.float32)
        output[name] = tensor

    return output


def processed_feature_collate(
    samples: Sequence[ProcessedFeatureSample],
    max_features_per_sample: int = 0,
    sequence_max_lens: Optional[Mapping[str, int]] = None,
    embedding_dims: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    """把结构化处理后的样本拼成 batch。

    输出同时包含：
    - 稀疏 token: feature_ids/feature_values/mask/labels
    - 序列特征: sequences[name][...]
    - 预训练 embedding: pretrained_embeddings[name] = Tensor[B, D]
    """

    if not samples:
        raise ValueError("samples 不能为空")

    sparse_samples = [
        LibSVMSample(sample.labels, sample.sparse.feature_ids, sample.sparse.feature_values)
        for sample in samples
    ]
    batch = make_collate_fn(max_features_per_sample=max_features_per_sample)(sparse_samples)

    sequence_max_lens = sequence_max_lens or {}
    sequence_names = sorted({name for sample in samples for name in sample.sequences})
    batch["sequences"] = {
        name: sequence_collate(
            [sample.sequences.get(name, SequenceFeatureBundle([], [], {}, {})) for sample in samples],
            max_len=int(sequence_max_lens.get(name, 0)),
        )
        for name in sequence_names
    }
    batch["pretrained_embeddings"] = pretrained_embedding_collate(
        [sample.pretrained_embeddings for sample in samples],
        embedding_dims=embedding_dims,
    )
    return batch


def _split_labels(label_token: str, label_separator: str) -> List[float]:
    """解析单目标或多目标 label。

    支持如下形式：
    - 单目标：`1 12:0.5 45:1`
    - 多目标：`1,0,1 12:0.5 45:1`
    """

    if label_separator and label_separator in label_token:
        return [float(x) for x in label_token.split(label_separator) if x != ""]
    return [float(label_token)]


def _parse_sample_metadata(comment: str) -> Dict[str, str]:
    """解析 libsvm 行尾注释中的样本元信息。

    支持格式：
    - `# traceid=t1 userid=u1`
    - `# trace_id=t1 user_id=u1`
    - `# traceid:t1 userid:u1`
    """

    metadata: Dict[str, str] = {}
    for token in comment.strip().split():
        if "=" in token:
            key, value = token.split("=", 1)
        elif ":" in token:
            key, value = token.split(":", 1)
        else:
            continue
        key = key.strip().lower()
        if key in {"traceid", "trace_id", "request_id", "req_id"}:
            metadata["trace_id"] = value.strip()
        elif key in {"userid", "user_id", "uid"}:
            metadata["user_id"] = value.strip()
    return metadata


def parse_libsvm_line(
    line: str,
    label_separator: str = ",",
    feature_id_offset: int = 1,
) -> Optional[LibSVMSample]:
    """解析一行 libsvm 文本。

    Args:
        line: 原始文本行。允许 `#` 后跟注释。
        label_separator: 多目标标签分隔符，默认逗号。
        feature_id_offset: 特征 ID 偏移量。默认 +1，把 0 留给 padding。

    Returns:
        解析成功返回 LibSVMSample；空行或纯注释返回 None。
    """

    # libsvm 生态中 `#` 后面常用于样本说明；这里额外支持在注释中放
    # traceid/userid，便于计算 GAUC/UAUC。
    feature_part, _, comment_part = line.partition("#")
    metadata = _parse_sample_metadata(comment_part)
    line = feature_part.strip()
    if not line:
        return None

    parts = line.split()
    labels = _split_labels(parts[0], label_separator)
    feature_ids: List[int] = []
    feature_values: List[float] = []

    for token in parts[1:]:
        # 兼容标准 libsvm ranking 样本中的 qid 字段，默认视作 trace_id。
        if token.startswith("qid:"):
            metadata.setdefault("trace_id", token.split(":", 1)[1])
            continue
        if ":" in token:
            fid_text, value_text = token.split(":", 1)
            value = float(value_text)
        else:
            fid_text, value = token, 1.0

        fid = int(fid_text) + feature_id_offset
        if fid <= 0:
            raise ValueError(f"feature_id + offset 必须大于 0，当前 token={token}")

        # value 为 0 的稀疏特征没有实际贡献，直接跳过可以节省计算。
        if value != 0.0:
            feature_ids.append(fid)
            feature_values.append(value)

    return LibSVMSample(
        labels=labels,
        feature_ids=feature_ids,
        feature_values=feature_values,
        trace_id=metadata.get("trace_id", ""),
        user_id=metadata.get("user_id", ""),
    )


class LibSVMDataset(Dataset):
    """libsvm 文件数据集，支持简单正负采样和样本数截断。"""

    def __init__(
        self,
        path: str | Path,
        label_separator: str = ",",
        sample_rate: float = 1.0,
        negative_sample_rate: float = 1.0,
        max_samples: Optional[int] = None,
        seed: int = 2026,
    ) -> None:
        self.path = Path(path)
        self.label_separator = label_separator
        self.samples: List[LibSVMSample] = []

        if not self.path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.path}")

        rng = random.Random(seed)
        for line in self.path.open("r", encoding="utf-8"):
            sample = parse_libsvm_line(line, label_separator=label_separator)
            if sample is None:
                continue

            # 全局采样：快速缩小训练集，常用于本地调试。
            if sample_rate < 1.0 and rng.random() > sample_rate:
                continue

            # 负样本采样：默认按第一个目标判断正负，多目标训练时通常第一个目标
            # 是主目标，例如 click。
            if sample.labels and sample.labels[0] <= 0.0:
                if negative_sample_rate < 1.0 and rng.random() > negative_sample_rate:
                    continue

            self.samples.append(sample)
            if max_samples is not None and len(self.samples) >= max_samples:
                break

        if not self.samples:
            raise ValueError(f"没有从 {self.path} 读取到有效样本")

        label_dim = len(self.samples[0].labels)
        for sample in self.samples:
            if len(sample.labels) != label_dim:
                raise ValueError("同一个数据文件内 label 维度不一致，请检查多目标标签格式")

        self.info = DataInfo(
            num_features=max((max(s.feature_ids) if s.feature_ids else 0) for s in self.samples) + 1,
            label_dim=label_dim,
            max_nnz=max(len(s.feature_ids) for s in self.samples),
            num_samples=len(self.samples),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> LibSVMSample:
        return self.samples[index]


def merge_data_info(infos: Iterable[DataInfo]) -> DataInfo:
    """合并 train/valid/test 的元信息，保证模型 embedding 表足够覆盖所有切分。"""

    info_list = list(infos)
    return DataInfo(
        num_features=max(info.num_features for info in info_list),
        label_dim=max(info.label_dim for info in info_list),
        max_nnz=max(info.max_nnz for info in info_list),
        num_samples=sum(info.num_samples for info in info_list),
    )


def make_collate_fn(max_features_per_sample: int = 0):
    """构造 batch 拼接函数。

    libsvm 样本是变长稀疏特征，本函数会把一个 batch padding 到相同长度。
    max_features_per_sample > 0 时会截断超长样本，避免偶发极长样本拖慢训练。
    """

    def collate(samples: Sequence[LibSVMSample]) -> Dict[str, torch.Tensor]:
        batch_size = len(samples)
        label_dim = len(samples[0].labels)
        batch_max_nnz = max(len(sample.feature_ids) for sample in samples)
        if max_features_per_sample > 0:
            batch_max_nnz = min(batch_max_nnz, max_features_per_sample)
        batch_max_nnz = max(batch_max_nnz, 1)

        feature_ids = torch.zeros(batch_size, batch_max_nnz, dtype=torch.long)
        feature_values = torch.zeros(batch_size, batch_max_nnz, dtype=torch.float32)
        mask = torch.zeros(batch_size, batch_max_nnz, dtype=torch.bool)
        labels = torch.zeros(batch_size, label_dim, dtype=torch.float32)
        trace_ids: List[str] = []
        user_ids: List[str] = []

        for row, sample in enumerate(samples):
            nnz = min(len(sample.feature_ids), batch_max_nnz)
            if nnz > 0:
                feature_ids[row, :nnz] = torch.tensor(sample.feature_ids[:nnz], dtype=torch.long)
                feature_values[row, :nnz] = torch.tensor(sample.feature_values[:nnz], dtype=torch.float32)
                mask[row, :nnz] = True
            labels[row] = torch.tensor(sample.labels, dtype=torch.float32)
            trace_ids.append(sample.trace_id)
            user_ids.append(sample.user_id)

        return {
            "feature_ids": feature_ids,
            "feature_values": feature_values,
            "mask": mask,
            "labels": labels,
            "trace_ids": trace_ids,
            "user_ids": user_ids,
        }

    return collate


def build_dataloaders(config: Dict) -> Tuple[Dict[str, DataLoader], DataInfo]:
    """根据配置创建 train/valid/test DataLoader。"""

    data_cfg = config.get("data", {})
    loader_cfg = config.get("loader", {})
    label_separator = data_cfg.get("label_separator", ",")
    max_features_per_sample = int(data_cfg.get("max_features_per_sample", 0))

    datasets: Dict[str, LibSVMDataset] = {}
    for split in ("train", "valid", "test"):
        path = data_cfg.get(f"{split}_path")
        if not path:
            continue
        prepared_path = prepare_input_path(path, config)
        negative_sample_rate = (
            float(data_cfg.get("negative_sample_rate", 1.0))
            if split == "train"
            else float(data_cfg.get(f"{split}_negative_sample_rate", 1.0))
        )

        datasets[split] = LibSVMDataset(
            path=prepared_path,
            label_separator=label_separator,
            sample_rate=float(data_cfg.get(f"{split}_sample_rate", 1.0)),
            negative_sample_rate=negative_sample_rate,
            max_samples=data_cfg.get(f"{split}_max_samples"),
            seed=int(config.get("seed", 2026)),
        )

    if "train" not in datasets:
        raise ValueError("配置中必须提供 data.train_path")

    data_info = merge_data_info(dataset.info for dataset in datasets.values())
    collate_fn = make_collate_fn(max_features_per_sample=max_features_per_sample)

    dataloaders: Dict[str, DataLoader] = {}
    for split, dataset in datasets.items():
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=int(loader_cfg.get("batch_size", 1024)),
            shuffle=(split == "train"),
            num_workers=int(loader_cfg.get("num_workers", 0)),
            pin_memory=bool(loader_cfg.get("pin_memory", True)),
            drop_last=bool(loader_cfg.get("drop_last", False)) if split == "train" else False,
            collate_fn=collate_fn,
            persistent_workers=bool(loader_cfg.get("persistent_workers", False))
            and int(loader_cfg.get("num_workers", 0)) > 0,
        )

    return dataloaders, data_info
