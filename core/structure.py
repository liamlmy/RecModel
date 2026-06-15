"""推荐模型常用结构层。

这里放和具体模型解耦的模块，DeepFM、DIN、MMOE、PEPNet、RankMixer 等模型
可以按需复用。实现上尽量保持输入/输出形状清晰，便于在生产代码中组装。
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def get_activation(name: str) -> nn.Module:
    """按名称创建激活函数。"""

    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "dice":
        return Dice()
    raise ValueError(f"不支持的激活函数: {name}")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization。

    RMSNorm 不减均值，只按 RMS 归一化，参数量和计算量都略低于 LayerNorm，
    在一些大模型和推荐 DNN tower 中常用于替代 LayerNorm。
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


def get_norm_layer(norm_type: str, dim: int) -> nn.Module:
    """按名称创建归一化层。

    支持：
    - none/identity: 不做归一化
    - batchnorm/bn: BatchNorm1d，适合 [B, D] DNN hidden
    - layernorm/ln: LayerNorm
    - rmsnorm/rms: RMSNorm
    """

    norm_type = (norm_type or "none").lower()
    if norm_type in {"none", "identity", "no", "false"}:
        return nn.Identity()
    if norm_type in {"batchnorm", "batch_norm", "bn"}:
        return nn.BatchNorm1d(dim)
    if norm_type in {"layernorm", "layer_norm", "ln"}:
        return nn.LayerNorm(dim)
    if norm_type in {"rmsnorm", "rms_norm", "rms"}:
        return RMSNorm(dim)
    raise ValueError(f"不支持的归一化类型: {norm_type}")


class Dice(nn.Module):
    """DIN 常用 Dice 激活函数。

    Dice 会根据 batch 内分布自适应调整激活门控，推荐系统中常比固定 ReLU 更稳。
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, keepdim=True, unbiased=False)
        p = torch.sigmoid((x - mean) / torch.sqrt(var + self.eps))
        return p * x + (1.0 - p) * self.alpha * x


class MLP(nn.Module):
    """通用多层感知机。"""

    def __init__(
        self,
        input_dim: int,
        hidden_units: Sequence[int],
        output_dim: Optional[int] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        norm_type: str = "none",
    ) -> None:
        super().__init__()
        if use_batch_norm and norm_type in {"", "none", None}:
            norm_type = "batchnorm"
        dims = [input_dim, *hidden_units]
        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            norm_layer = get_norm_layer(norm_type, out_dim)
            if not isinstance(norm_layer, nn.Identity):
                layers.append(norm_layer)
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        if output_dim is not None:
            last_dim = dims[-1] if hidden_units else input_dim
            layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Gate(nn.Module):
    """通用门控层，用于特征选择、专家加权或残差调制。"""

    def __init__(self, input_dim: int, output_dim: int, hidden_units: Sequence[int] = ()) -> None:
        super().__init__()
        self.net = MLP(input_dim, hidden_units, output_dim=output_dim, activation="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class ScaledDotProductAttention(nn.Module):
    """传统 scaled dot-product attention。

    use_flash_attention=True 时优先走 PyTorch 2.x 的
    scaled_dot_product_attention。是否真的启用 FlashAttention kernel 由 PyTorch、
    CUDA、GPU 架构和 dtype 共同决定；不满足条件时 PyTorch 会自动回退。
    """

    def __init__(self, dropout: float = 0.0, use_flash_attention: bool = False) -> None:
        super().__init__()
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_flash_attention:
            attn_mask = None
            if mask is not None:
                # SDPA 的 bool mask 中 True 表示参与 attention。
                attn_mask = mask[:, None, None, :].to(torch.bool)
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )

        scale = query.size(-1) ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return torch.matmul(weights, value)


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力，封装传统 attention 与可选 FlashAttention。"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn = ScaledDotProductAttention(dropout=dropout, use_flash_attention=use_flash_attention)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = self.attn(q, k, v, mask=mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out(out)


class DINAttentionLayer(nn.Module):
    """DIN 的 activation unit。

    Args:
        query: 候选物品/广告 embedding，形状 [B, D]。
        keys: 用户历史行为 embedding，形状 [B, T, D]。
        mask: 历史行为有效位，形状 [B, T]。
    """

    def __init__(self, embed_dim: int, hidden_units: Sequence[int] = (80, 40), activation: str = "dice") -> None:
        super().__init__()
        self.local_mlp = MLP(
            input_dim=embed_dim * 4,
            hidden_units=hidden_units,
            output_dim=1,
            activation=activation,
        )

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len = keys.size(1)
        query_expand = query.unsqueeze(1).expand(-1, seq_len, -1)
        attn_input = torch.cat([query_expand, keys, query_expand - keys, query_expand * keys], dim=-1)
        scores = self.local_mlp(attn_input.reshape(-1, attn_input.size(-1))).view(keys.size(0), seq_len)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return torch.sum(weights * keys, dim=1)


class SENetLayer(nn.Module):
    """SENet 特征重标定层。

    输入是 field/稀疏 token 级 embedding [B, F, D]，输出同形状。这里按 field
    维做 squeeze，让模型学习不同特征位的重要性。
    """

    def __init__(self, field_size: int, reduction_ratio: int = 3) -> None:
        super().__init__()
        reduced_size = max(1, field_size // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(field_size, reduced_size),
            nn.ReLU(),
            nn.Linear(reduced_size, field_size),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        squeeze = x.mean(dim=-1)
        if mask is not None:
            squeeze = squeeze.masked_fill(~mask, 0.0)
        weights = self.excitation(squeeze).unsqueeze(-1)
        return x * weights


class LHUC(nn.Module):
    """Learning Hidden Unit Contributions。

    LHUC 根据上下文生成 0~2 的缩放系数，对 hidden units 做个性化调制。
    """

    def __init__(self, context_dim: int, hidden_dim: int, hidden_units: Sequence[int] = (64,)) -> None:
        super().__init__()
        self.gate = MLP(context_dim, hidden_units, output_dim=hidden_dim, activation="relu")

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        scale = 2.0 * torch.sigmoid(self.gate(context))
        return hidden * scale


class MMOE(nn.Module):
    """Multi-gate Mixture-of-Experts，多目标推荐常用模块。"""

    def __init__(
        self,
        input_dim: int,
        expert_hidden_units: Sequence[int],
        expert_output_dim: int,
        num_experts: int,
        num_tasks: int,
        dropout: float = 0.0,
        norm_type: str = "none",
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [
                MLP(input_dim, expert_hidden_units, output_dim=expert_output_dim, dropout=dropout, norm_type=norm_type)
                for _ in range(num_experts)
            ]
        )
        self.gates = nn.ModuleList([nn.Linear(input_dim, num_experts) for _ in range(num_tasks)])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        task_outputs: List[torch.Tensor] = []
        for gate in self.gates:
            weights = torch.softmax(gate(x), dim=-1).unsqueeze(-1)
            task_outputs.append(torch.sum(weights * expert_outputs, dim=1))
        return task_outputs


class PEPNetGate(nn.Module):
    """PEPNet 风格的个性化门控。

    该模块可以作为 EPNet/PPNet 的基础组件：用 context 生成和 target 等长的
    门控向量，对 embedding 或 tower hidden 做逐维调制。
    """

    def __init__(self, context_dim: int, target_dim: int, hidden_units: Sequence[int] = (64,)) -> None:
        super().__init__()
        self.gate = MLP(context_dim, hidden_units, output_dim=target_dim, activation="relu")

    def forward(self, target: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        scale = 2.0 * torch.sigmoid(self.gate(context))
        return target * scale


class RankMixerBlock(nn.Module):
    """RankMixer/MLP-Mixer 风格的排序特征混合块。

    推荐排序模型通常需要同时建模“特征位之间的交互”和“embedding 通道之间的
    交互”。该模块先沿 feature/token 维混合，再沿 channel 维混合，并使用残差
    保持训练稳定。
    """

    def __init__(
        self,
        field_size: int,
        embed_dim: int,
        token_mlp_dim: int,
        channel_mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(embed_dim)
        self.token_mixer = nn.Sequential(
            nn.Linear(field_size, token_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_mlp_dim, field_size),
            nn.Dropout(dropout),
        )
        self.channel_norm = nn.LayerNorm(embed_dim)
        self.channel_mixer = nn.Sequential(
            nn.Linear(embed_dim, channel_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        y = self.token_norm(x).transpose(1, 2)
        y = self.token_mixer(y).transpose(1, 2)
        x = x + y
        x = x + self.channel_mixer(self.channel_norm(x))
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        return x


class RankMixer(nn.Module):
    """多个 RankMixerBlock 堆叠。"""

    def __init__(
        self,
        field_size: int,
        embed_dim: int,
        num_layers: int = 2,
        token_mlp_dim: int = 64,
        channel_mlp_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                RankMixerBlock(field_size, embed_dim, token_mlp_dim, channel_mlp_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask=mask)
        return x
