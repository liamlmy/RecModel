"""推荐模型常用结构层。

这里放和具体模型解耦的模块，DeepFM、DIN、MMOE、PEPNet、RankMixer 等模型
可以按需复用。实现上尽量保持输入/输出形状清晰，便于在生产代码中组装。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from core.data import FeatureSpec


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


class APGLayer(nn.Module):
    """Adaptive Parameter Generation 动态参数层。

    参考 APG 论文中的低秩参数分解思路，对每个样本根据 condition 生成中心矩阵：

        W_b = U @ S_b @ V

    其中 U/V 是共享参数，S_b 由生成网络按样本生成。forward 中直接按
    `x @ U @ S_b @ V` 计算，避免显式构造完整的 batch 级权重矩阵。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        condition_dim: Optional[int] = None,
        rank: int = 16,
        generator_hidden_units: Sequence[int] = (64,),
        activation: str = "relu",
        dropout: float = 0.0,
        norm_type: str = "none",
        bias: bool = True,
        over_parameter_dim: Optional[int] = None,
        init_center_identity: bool = True,
        center_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim 和 output_dim 必须为正数")
        if rank <= 0:
            raise ValueError("rank 必须为正数")
        if over_parameter_dim is not None and over_parameter_dim <= 0:
            raise ValueError("over_parameter_dim 必须为正数")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.condition_dim = int(condition_dim or input_dim)
        self.rank = int(rank)
        self.over_parameter_dim = int(over_parameter_dim) if over_parameter_dim is not None else None
        self.center_scale = float(center_scale)

        if self.over_parameter_dim is None:
            self.left_factor = nn.Parameter(torch.empty(self.input_dim, self.rank))
            self.right_factor = nn.Parameter(torch.empty(self.rank, self.output_dim))
        else:
            self.left_factor_a = nn.Parameter(torch.empty(self.input_dim, self.over_parameter_dim))
            self.left_factor_b = nn.Parameter(torch.empty(self.over_parameter_dim, self.rank))
            self.right_factor_a = nn.Parameter(torch.empty(self.rank, self.over_parameter_dim))
            self.right_factor_b = nn.Parameter(torch.empty(self.over_parameter_dim, self.output_dim))

        self.generator = MLP(
            self.condition_dim,
            generator_hidden_units,
            output_dim=self.rank * self.rank,
            activation=activation,
            dropout=dropout,
            norm_type=norm_type,
        )
        self.bias = nn.Parameter(torch.zeros(self.output_dim)) if bias else None
        self.reset_parameters(init_center_identity=init_center_identity)

    def reset_parameters(self, init_center_identity: bool = True) -> None:
        if self.over_parameter_dim is None:
            nn.init.xavier_uniform_(self.left_factor)
            nn.init.xavier_uniform_(self.right_factor)
        else:
            nn.init.xavier_uniform_(self.left_factor_a)
            nn.init.xavier_uniform_(self.left_factor_b)
            nn.init.xavier_uniform_(self.right_factor_a)
            nn.init.xavier_uniform_(self.right_factor_b)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        final_linear = self._generator_final_linear()
        if final_linear is None:
            return
        if init_center_identity:
            nn.init.normal_(final_linear.weight, mean=0.0, std=1e-3)
            center_bias = torch.eye(self.rank).flatten() * self.center_scale
            with torch.no_grad():
                final_linear.bias.copy_(center_bias)
        else:
            nn.init.xavier_uniform_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

    def _generator_final_linear(self) -> Optional[nn.Linear]:
        for module in reversed(self.generator.net):
            if isinstance(module, nn.Linear):
                return module
        return None

    def shared_factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回共享的左右低秩因子 U/V。"""

        if self.over_parameter_dim is None:
            return self.left_factor, self.right_factor
        left = torch.matmul(self.left_factor_a, self.left_factor_b)
        right = torch.matmul(self.right_factor_a, self.right_factor_b)
        return left, right

    def generate_center(self, condition: torch.Tensor) -> torch.Tensor:
        """根据 condition 生成 batch 级中心矩阵 S_b，形状 [B, rank, rank]。"""

        if condition.dim() != 2:
            raise ValueError("condition 必须是 [B, condition_dim] 的二维张量")
        if condition.size(-1) != self.condition_dim:
            raise ValueError(
                f"condition 最后一维应为 {self.condition_dim}，当前为 {condition.size(-1)}"
            )
        return self.generator(condition).view(condition.size(0), self.rank, self.rank)

    def generated_weight(self, condition: torch.Tensor) -> torch.Tensor:
        """显式生成完整权重矩阵，主要用于调试或可视化，形状 [B, input_dim, output_dim]。"""

        left, right = self.shared_factors()
        center = self.generate_center(condition)
        return torch.einsum("ir,brs,so->bio", left, center, right)

    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("x 必须是 [B, input_dim] 的二维张量")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"x 最后一维应为 {self.input_dim}，当前为 {x.size(-1)}")
        if condition is None:
            condition = x
        if condition.size(0) != x.size(0):
            raise ValueError("condition 与 x 的 batch size 必须一致")

        left, right = self.shared_factors()
        center = self.generate_center(condition)
        hidden = torch.matmul(x, left)
        hidden = torch.bmm(hidden.unsqueeze(1), center).squeeze(1)
        output = torch.matmul(hidden, right)
        if self.bias is not None:
            output = output + self.bias
        return output


class FMInteraction(nn.Module):
    """FM 二阶交叉项。

    输入 field embedding，按公式 0.5 * ((sum x)^2 - sum(x^2)) 计算二阶交叉。
    默认返回 [B, D] 的交叉向量；需要标量二阶项时可设置 reduce_sum=True。
    """

    def __init__(self, reduce_sum: bool = False) -> None:
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if embeddings.dim() != 3:
            raise ValueError("embeddings 必须是 [B, F, D] 的三维张量")
        if mask is not None:
            embeddings = embeddings.masked_fill(~mask.unsqueeze(-1), 0.0)
        square_of_sum = torch.sum(embeddings, dim=1).pow(2)
        sum_of_square = torch.sum(embeddings.pow(2), dim=1)
        interactions = 0.5 * (square_of_sum - sum_of_square)
        if self.reduce_sum:
            return interactions.sum(dim=-1, keepdim=True)
        return interactions


class CrossNetwork(nn.Module):
    """DCN v1 Cross Network。

    每层计算：

        x_{l+1} = x_0 * (x_l w_l) + b_l + x_l

    其中 w_l 是向量参数，因此每层参数量为 O(D)。
    """

    def __init__(self, input_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim 必须为正数")
        if num_layers <= 0:
            raise ValueError("num_layers 必须为正数")
        self.input_dim = int(input_dim)
        self.num_layers = int(num_layers)
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.empty(self.input_dim)) for _ in range(self.num_layers)]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.input_dim)) for _ in range(self.num_layers)]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.weights:
            nn.init.xavier_uniform_(weight.view(1, -1))
        for bias in self.biases:
            nn.init.zeros_(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("x 必须是 [B, input_dim] 的二维张量")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"x 最后一维应为 {self.input_dim}，当前为 {x.size(-1)}")
        x0 = x
        xl = x
        for weight, bias in zip(self.weights, self.biases):
            cross = torch.matmul(xl, weight).unsqueeze(-1)
            xl = x0 * cross + bias + xl
        return xl


class DCNV2CrossLayer(nn.Module):
    """DCN-V2 单层矩阵交叉层。

    full-rank 模式使用完整矩阵 W；low-rank 模式使用 U/V 分解，降低参数量。
    """

    def __init__(
        self,
        input_dim: int,
        low_rank: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim 必须为正数")
        if low_rank is not None and low_rank <= 0:
            raise ValueError("low_rank 必须为正数")
        self.input_dim = int(input_dim)
        self.low_rank = int(low_rank) if low_rank is not None else None
        if self.low_rank is None:
            self.weight = nn.Parameter(torch.empty(self.input_dim, self.input_dim))
        else:
            self.u = nn.Parameter(torch.empty(self.input_dim, self.low_rank))
            self.v = nn.Parameter(torch.empty(self.low_rank, self.input_dim))
        self.bias = nn.Parameter(torch.zeros(self.input_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.low_rank is None:
            nn.init.xavier_uniform_(self.weight)
        else:
            nn.init.xavier_uniform_(self.u)
            nn.init.xavier_uniform_(self.v)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        if self.low_rank is None:
            projection = torch.matmul(xl, self.weight)
        else:
            projection = torch.matmul(torch.matmul(xl, self.u), self.v)
        if self.bias is not None:
            projection = projection + self.bias
        return x0 * projection + xl


class DCNV2CrossNetwork(nn.Module):
    """DCN-V2 Cross Network。

    每层使用矩阵参数建模特征交叉：

        x_{l+1} = x_0 * (W_l x_l + b_l) + x_l

    low_rank 不为空时，W_l 由 U_l @ V_l 近似。
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 2,
        low_rank: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers 必须为正数")
        self.input_dim = int(input_dim)
        self.num_layers = int(num_layers)
        self.layers = nn.ModuleList(
            [DCNV2CrossLayer(input_dim, low_rank=low_rank, bias=bias) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("x 必须是 [B, input_dim] 的二维张量")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"x 最后一维应为 {self.input_dim}，当前为 {x.size(-1)}")
        x0 = x
        xl = x
        for layer in self.layers:
            xl = layer(x0, xl)
        return xl


class CINLayer(nn.Module):
    """xDeepFM 的 Compressed Interaction Network。

    输入 field embedding [B, F, D]，逐层显式建模 bounded-degree feature interaction。
    输出为每层交互结果在 embedding 维 sum pooling 后的拼接，形状 [B, output_dim]。
    """

    def __init__(
        self,
        field_size: int,
        layer_sizes: Sequence[int],
        split_half: bool = True,
        activation: Optional[str] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if field_size <= 0:
            raise ValueError("field_size 必须为正数")
        if not layer_sizes:
            raise ValueError("layer_sizes 不能为空")
        if any(size <= 0 for size in layer_sizes):
            raise ValueError("layer_sizes 中所有值都必须为正数")

        self.field_size = int(field_size)
        self.layer_sizes = [int(size) for size in layer_sizes]
        self.split_half = bool(split_half)
        self.activation = get_activation(activation) if activation else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        convs: List[nn.Module] = []
        prev_field_size = self.field_size
        output_dims: List[int] = []
        for idx, layer_size in enumerate(self.layer_sizes):
            convs.append(nn.Conv1d(prev_field_size * self.field_size, layer_size, kernel_size=1))
            is_last = idx == len(self.layer_sizes) - 1
            if self.split_half and not is_last:
                next_field_size = layer_size // 2
                output_dims.append(layer_size - next_field_size)
            else:
                next_field_size = layer_size
                output_dims.append(layer_size)
            if next_field_size <= 0:
                raise ValueError("split_half=True 时，中间层 layer_size 至少为 2")
            prev_field_size = next_field_size
        self.convs = nn.ModuleList(convs)
        self.output_dim = sum(output_dims)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("x 必须是 [B, F, D] 的三维张量")
        if x.size(1) != self.field_size:
            raise ValueError(f"x field 维应为 {self.field_size}，当前为 {x.size(1)}")
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        x0 = x
        hidden = x
        outputs: List[torch.Tensor] = []
        for idx, conv in enumerate(self.convs):
            interactions = torch.einsum("bhd,bmd->bhmd", hidden, x0)
            batch_size, hidden_fields, base_fields, embed_dim = interactions.shape
            interactions = interactions.reshape(batch_size, hidden_fields * base_fields, embed_dim)
            layer_output = conv(interactions)
            layer_output = self.dropout(self.activation(layer_output))

            is_last = idx == len(self.convs) - 1
            if self.split_half and not is_last:
                next_hidden, direct_output = torch.split(
                    layer_output,
                    [layer_output.size(1) // 2, layer_output.size(1) - layer_output.size(1) // 2],
                    dim=1,
                )
                hidden = next_hidden
                outputs.append(direct_output)
            else:
                hidden = layer_output
                outputs.append(layer_output)

        pooled_outputs = [item.sum(dim=-1) for item in outputs]
        return torch.cat(pooled_outputs, dim=1)


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


class StructuredFeatureInputLayer(nn.Module):
    """结构化推荐特征输入模块。

    处理规则：
    - dense 数值特征：原始值直接参与 concat，同时投影成 field embedding 供 SENet 使用。
    - one-hot 特征：查 8/16 维 embedding。
    - multi-hot 特征：查 8/16 维 embedding 后按权重 pooling。
    - sequence 特征：item ID + side info embedding 后，经 DIN 或 self-attention 输出 16/32 维向量。
    - pretrained embedding：直接 concat，同时可投影成 SENet field。
    """

    def __init__(
        self,
        num_features: int,
        feature_specs: Sequence[FeatureSpec],
        input_cfg: Dict,
    ) -> None:
        super().__init__()
        self.feature_specs = list(feature_specs)
        self.dense_specs = [spec for spec in self.feature_specs if spec.feature_type.lower() == "numeric"]
        self.one_hot_specs = [spec for spec in self.feature_specs if spec.feature_type.lower() == "one_hot"]
        self.multi_hot_specs = [spec for spec in self.feature_specs if spec.feature_type.lower() == "multi_hot"]
        self.sequence_specs = [spec for spec in self.feature_specs if spec.feature_type.lower() == "sequence"]
        self.pretrained_specs = [
            spec for spec in self.feature_specs if spec.feature_type.lower() == "pretrained_embedding"
        ]

        self.one_hot_dim = int(input_cfg.get("one_hot_embedding_dim", 16))
        self.multi_hot_dim = int(input_cfg.get("multi_hot_embedding_dim", 16))
        self.sequence_dim = int(input_cfg.get("sequence_embedding_dim", 32))
        self.senet_field_dim = int(input_cfg.get("senet_field_dim", 16))
        self.sequence_attention_type = str(input_cfg.get("sequence_attention_type", "din")).lower()
        self.use_input_senet = bool(input_cfg.get("use_senet", False))

        self.one_hot_embedding = nn.Embedding(num_features, self.one_hot_dim, padding_idx=0)
        self.multi_hot_embedding = nn.Embedding(num_features, self.multi_hot_dim, padding_idx=0)
        self.sequence_embedding = nn.Embedding(num_features, self.sequence_dim, padding_idx=0)
        self.sequence_side_embedding = nn.Embedding(num_features, self.sequence_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.one_hot_embedding.weight)
        nn.init.xavier_uniform_(self.multi_hot_embedding.weight)
        nn.init.xavier_uniform_(self.sequence_embedding.weight)
        nn.init.xavier_uniform_(self.sequence_side_embedding.weight)

        pretrained_dim = sum(max(1, int(spec.embedding_dim)) for spec in self.pretrained_specs)
        self.non_sequence_dim = (
            len(self.dense_specs)
            + len(self.one_hot_specs) * self.one_hot_dim
            + len(self.multi_hot_specs) * self.multi_hot_dim
            + pretrained_dim
        )
        self.raw_output_dim = self.non_sequence_dim + len(self.sequence_specs) * self.sequence_dim

        self.sequence_query = nn.Linear(max(1, self.non_sequence_dim), self.sequence_dim)
        if self.sequence_attention_type == "din":
            self.sequence_attentions = nn.ModuleDict(
                {spec.name: DINAttentionLayer(self.sequence_dim) for spec in self.sequence_specs}
            )
            self.sequence_self_attentions = nn.ModuleDict()
        elif self.sequence_attention_type in {"attention", "self_attention"}:
            self.sequence_attentions = nn.ModuleDict()
            self.sequence_self_attentions = nn.ModuleDict(
                {
                    spec.name: MultiHeadSelfAttention(
                        self.sequence_dim,
                        num_heads=int(input_cfg.get("sequence_attention_num_heads", 4)),
                        use_flash_attention=bool(input_cfg.get("use_flash_attention", False)),
                    )
                    for spec in self.sequence_specs
                }
            )
        else:
            raise ValueError("sequence_attention_type 仅支持 din/attention/self_attention")

        self.field_count = (
            len(self.dense_specs)
            + len(self.one_hot_specs)
            + len(self.multi_hot_specs)
            + len(self.sequence_specs)
            + len(self.pretrained_specs)
        )
        self.dense_projectors = nn.ModuleDict(
            {spec.name: nn.Linear(1, self.senet_field_dim) for spec in self.dense_specs}
        )
        self.one_hot_projectors = nn.ModuleDict(
            {spec.name: nn.Linear(self.one_hot_dim, self.senet_field_dim) for spec in self.one_hot_specs}
        )
        self.multi_hot_projectors = nn.ModuleDict(
            {spec.name: nn.Linear(self.multi_hot_dim, self.senet_field_dim) for spec in self.multi_hot_specs}
        )
        self.sequence_projectors = nn.ModuleDict(
            {spec.name: nn.Linear(self.sequence_dim, self.senet_field_dim) for spec in self.sequence_specs}
        )
        self.pretrained_projectors = nn.ModuleDict(
            {
                spec.name: nn.Linear(max(1, int(spec.embedding_dim)), self.senet_field_dim)
                for spec in self.pretrained_specs
            }
        )
        self.senet = SENetLayer(self.field_count) if self.use_input_senet and self.field_count > 0 else None
        self.output_dim = self.field_count * self.senet_field_dim if self.senet is not None else self.raw_output_dim

    def _pool_multi_hot(self, feature: Dict[str, torch.Tensor]) -> torch.Tensor:
        ids = feature["feature_ids"]
        values = feature["feature_values"].unsqueeze(-1)
        mask = feature["mask"].unsqueeze(-1)
        embeddings = self.multi_hot_embedding(ids) * values
        embeddings = embeddings.masked_fill(~mask, 0.0)
        denom = mask.sum(dim=1).clamp_min(1).to(embeddings.dtype)
        return embeddings.sum(dim=1) / denom

    def _sequence_repr(
        self,
        spec: FeatureSpec,
        sequence_batch: Dict[str, torch.Tensor],
        query_context: torch.Tensor,
    ) -> torch.Tensor:
        item_ids = sequence_batch["item_feature_ids"]
        item_values = sequence_batch["item_feature_values"].unsqueeze(-1)
        mask = sequence_batch["item_mask"]
        seq_embeddings = self.sequence_embedding(item_ids) * item_values
        for side_ids in sequence_batch.get("side_feature_ids", {}).values():
            seq_embeddings = seq_embeddings + self.sequence_side_embedding(side_ids)
        seq_embeddings = seq_embeddings.masked_fill(~mask.unsqueeze(-1), 0.0)

        if spec.name in self.sequence_attentions:
            query = self.sequence_query(query_context)
            return self.sequence_attentions[spec.name](query, seq_embeddings, mask)

        attended = self.sequence_self_attentions[spec.name](seq_embeddings, mask)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(attended.dtype)
        return attended.masked_fill(~mask.unsqueeze(-1), 0.0).sum(dim=1) / denom

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = batch["labels"].size(0)
        device = batch["labels"].device
        raw_vectors: List[torch.Tensor] = []
        field_vectors: List[torch.Tensor] = []

        dense_tensor = batch.get("dense_features")
        dense_names = batch.get("dense_feature_names", [])
        dense_by_name = {}
        if dense_tensor is not None:
            dense_by_name = {name: dense_tensor[:, idx : idx + 1] for idx, name in enumerate(dense_names)}
        for spec in self.dense_specs:
            value = dense_by_name.get(spec.name, torch.zeros(batch_size, 1, device=device))
            raw_vectors.append(value)
            field_vectors.append(self.dense_projectors[spec.name](value))

        for spec in self.one_hot_specs:
            ids = batch.get("one_hot_feature_ids", {}).get(spec.name)
            if ids is None:
                ids = torch.zeros(batch_size, dtype=torch.long, device=device)
            vector = self.one_hot_embedding(ids)
            raw_vectors.append(vector)
            field_vectors.append(self.one_hot_projectors[spec.name](vector))

        for spec in self.multi_hot_specs:
            feature = batch.get("multi_hot_features", {}).get(spec.name)
            if feature is None:
                vector = torch.zeros(batch_size, self.multi_hot_dim, device=device)
            else:
                vector = self._pool_multi_hot(feature)
            raw_vectors.append(vector)
            field_vectors.append(self.multi_hot_projectors[spec.name](vector))

        for spec in self.pretrained_specs:
            vector = batch.get("pretrained_embeddings", {}).get(spec.name)
            expected_dim = max(1, int(spec.embedding_dim))
            if vector is None:
                vector = torch.zeros(batch_size, expected_dim, device=device)
            raw_vectors.append(vector)
            field_vectors.append(self.pretrained_projectors[spec.name](vector))

        query_context = torch.cat(raw_vectors, dim=-1) if raw_vectors else torch.zeros(batch_size, 1, device=device)
        for spec in self.sequence_specs:
            sequence_batch = batch.get("sequences", {}).get(spec.name)
            if sequence_batch is None:
                vector = torch.zeros(batch_size, self.sequence_dim, device=device)
            else:
                vector = self._sequence_repr(spec, sequence_batch, query_context)
            raw_vectors.append(vector)
            field_vectors.append(self.sequence_projectors[spec.name](vector))

        if self.senet is not None:
            field_tensor = torch.stack(field_vectors, dim=1)
            field_tensor = self.senet(field_tensor)
            return field_tensor.flatten(start_dim=1)
        return torch.cat(raw_vectors, dim=-1)
