# 精排模型训练工程

这是一个面向排序场景的训练模板，支持 libsvm 与 JSONL 数据、多目标/单目标训练、混合精度、可选 FlashAttention 结构层，以及 MMOE、DIN Attention、PEPNet、SENet、LHUC、RankMixer 等常见模块。

## 环境配置

适配 NVIDIA L20 GPU 的推荐组合：

- Python 3.12
- PyTorch 2.11.0
- CUDA 12.8 wheel
- NVIDIA Driver 建议 570+

安装示例：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install PyYAML tensorboard
```

如果服务器驱动只适配 CUDA 12.6，可以改用 PyTorch 官方 cu126 wheel。

项目已在 `pyproject.toml` 中声明 `requires-python = ">=3.12,<3.13"`。如果你使用
Conda，建议创建独立环境：

```bash
conda create -n deepfm-py312 python=3.12 -y
conda activate deepfm-py312
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install PyYAML tensorboard
```

## 项目结构

```text
.
├── core/main.py         # 读取配置、训练、验证、测试、导出
├── core/model.py        # DeepFM 模型、输入层、FM 层、DNN/多任务输出层
├── core/structure.py    # DIN Attention、MMOE、PEPNet、Gate、Attention、RankMixer、SENet、LHUC
├── core/data.py         # libsvm 解析、采样、label/feature 识别、DataLoader
├── conf/common.yaml     # 数据路径、路径类型、训练、导出、debug 等通用配置
├── conf/model.yaml      # 模型配置，当前包含结构化 input + SENet + MMOE 示例
├── data/*.libsvm        # 可直接运行的小样本数据
├── checkpoints/         # 训练 checkpoint 输出目录
└── exports/             # TorchScript 导出目录
```

## 数据格式

单目标 libsvm：

```text
1 12:1 35:0.8 98:1
0 11:1 39:0.2 76:1
```

多目标 libsvm：

```text
1,0 12:1 35:0.8 98:1
0,1 11:1 39:0.2 76:1
```

默认用逗号分隔多目标 label，可在 `conf/common.yaml` 中修改：

```yaml
data:
  label_separator: ","
  path_type: local        # local 或 hdfs
  train_path: data/train.libsvm
model:
  task_names: ["click", "convert"]
```

特征 ID 会自动整体加 1，把 ID 0 保留为 padding。`num_features` 默认自动从 train/valid/test 推断。

如果需要计算 GAUC/UAUC，可以在 libsvm 行尾注释中携带分组键：

```text
1,0 12:1 35:0.8 98:1 # traceid=req_001 userid=user_123
```

训练日志会输出并写入 TensorBoard：

- `auc`：整体 AUC
- `gauc`：按 `traceid` 分组加权平均 AUC
- `uauc`：按 `userid` 分组加权平均 AUC
- 多目标场景下也会记录 `click_auc`、`click_gauc`、`click_uauc` 等任务级指标

## 结构化特征样本

除 libsvm 外，项目还提供了一套 JSONL 结构化样本，用于覆盖更贴近工业排序的
复杂特征：

- `data/rich_train.jsonl`：24 条训练样本
- `data/rich_valid.jsonl`：8 条验证样本
- `data/rich_test.jsonl`：8 条测试样本
- `data/rich_feature_schema.json`：字段 schema 和 `FeatureSpec` 配置

每条 rich 样本包含：

- 20 个数值特征：`num_001` ~ `num_020`
- 10 个 one-hot 特征：`oh_001` ~ `oh_010`
- 5 个 multi-hot 特征：`mh_001` ~ `mh_005`
- 3 个序列特征：`seq_001` ~ `seq_003`，每个 item 带 `cate`、`brand` side info
- 2 个预训练纯 embedding 特征：`pretrained_user_emb`、`pretrained_item_emb`

结构化样本可用 `RawFeatureDataset`、`build_feature_specs` 和
`processed_feature_collate` 读取。预训练 embedding 不进入 ID embedding 表，
会以 `batch["pretrained_embeddings"][name] = Tensor[B, D]` 的形式输出。

结构化模型链路可以通过 `model.model_type: mymodel` 开启：

- dense 数值特征：原始值直接拼接，同时投影成 SENet field。
- one-hot 特征：查 8 或 16 维 embedding。
- multi-hot 特征：查 8 或 16 维 embedding 后按 mask/权重 pooling。
- 序列特征：主 ID 与 side info 共同 embedding，通过 DIN 或 self-attention 输出 16 或 32 维向量。
- 预训练纯 embedding 特征：不查表，直接 concat。
- input 模块之后可选 `model.input.use_senet: true`，SENet 输出再进入 MMOE 和各任务 tower。

直接运行结构化 input 模块示例：

```bash
python -m core.main --mode train
```

## 训练

```bash
python -m core.main --mode train
```

训练过程会保存：

- `checkpoints/latest.pt`
- `checkpoints/best.pt`
- `exports/deepfm_traced.pt`
- `exports/config.json`

## 验证、测试、导出

```bash
python -m core.main --mode valid --checkpoint checkpoints/best.pt
python -m core.main --mode test --checkpoint checkpoints/best.pt
python -m core.main --mode export --checkpoint checkpoints/best.pt
```

默认会合并：

```bash
conf/common.yaml + conf/model.yaml
```

也可以指定配置：

```bash
python -m core.main --common-config conf/common.yaml --model-config conf/model.yaml --mode train
python -m core.main --config conf/experiment_override.yaml --mode train
```

`--config` 是可选覆盖配置，会在 `common + model` 之后合并，适合临时实验。

## 配置拆分

`conf/model.yaml` 放当前模型实验的覆盖配置；结构化 MMOE 示例已经合并到这里，
包括 rich JSONL 数据路径、checkpoint/export/log 目录，以及 input/MMOE/tower 等模型结构。

`conf/common.yaml` 放非模型结构配置，包括：

- `storage`：输入/输出路径类型，支持 `local`、`hdfs` 声明
- `data`：train/valid/test 输入路径、label 分隔符、采样配置
- `loader`：batch size、num workers、pin memory
- `train`：学习率、epoch、checkpoint、AMP、loss、warmup
- `export`：模型导出目录
- `debug`：TensorBoard 开关和日志目录

HDFS 输入示例：

```yaml
storage:
  input_type: hdfs
  hdfs_cmd: hdfs
  local_cache_dir: .cache/hdfs

data:
  path_type: hdfs
  train_path: hdfs:///path/to/train.libsvm
  valid_path: hdfs:///path/to/valid.libsvm
  test_path: hdfs:///path/to/test.libsvm
```

## 常用配置说明

```yaml
model:
  model_type: "deepfm"         # deepfm 或 mymodel
  dnn_input_mode: "pool"      # pool 更适合变长 libsvm；flatten 更接近传统固定 field DeepFM
  norm_type: "none"           # none、batchnorm、layernorm、rmsnorm
  use_mmoe: true              # 多目标训练建议开启
  use_senet: true             # 对稀疏特征位做重标定
  use_rankmixer: false        # 特征位和 embedding 通道混合
  use_lhuc: false             # hidden unit 个性化调制
  use_pepnet: false           # PEPNet 风格门控
  use_self_attention: false   # 在 DeepFM 的 embedding 序列上增加多头自注意力
  use_flash_attention: false  # use_self_attention=true 时可尝试启用 PyTorch SDPA flash kernel
  stop_gradient: []           # 例如 ["fm_vector", "dnn_input"]，用于局部梯度截断
  input:
    one_hot_embedding_dim: 16
    multi_hot_embedding_dim: 16
    sequence_embedding_dim: 32
    sequence_attention_type: "din"
    sequence_attention_num_heads: 4
    use_flash_attention: false
    use_senet: false
    senet_field_dim: 16

train:
  use_amp: true
  amp_dtype: "bfloat16"       # L20/Ada 上通常推荐 bfloat16
  use_focal_loss: false       # true 时使用多目标 Sigmoid Focal Loss
  focal_gamma: 2.0
  focal_alpha:                # 例如 [0.25, 0.5]
  task_weights:               # 例如 [1.0, 2.0]
  loss_warmup:
    enabled: false
    steps: 100
    start_factor: 0.1
    end_factor: 1.0

debug:
  tensorboard:
    enabled: true              # false 时完全关闭 TensorBoard
    log_dir: runs/model
    log_interval: 20
    log_histograms: false
```

## 生产使用建议

- 大规模训练时，把 `core/data.py` 中 `LibSVMDataset` 改成 `IterableDataset`，避免一次性加载全量样本。
- 多目标任务默认使用 BCEWithLogitsLoss，可配置 `train.pos_weight`；长尾目标较难学时也可以开启 `train.use_focal_loss`，并设置 `focal_alpha`、`task_weights`。
- 多分支模型训练早期不稳定时，可以开启 `train.loss_warmup`，让反传 loss 从较小系数线性升到 1。
- 做消融或冻结局部结构输出时，可以配置 `model.stop_gradient`，对指定结构输出执行 `detach()`。
- TensorBoard 默认写入 `runs/model`，启动方式：`tensorboard --logdir runs --port 6006`。
- L20 上建议优先尝试 `amp_dtype: bfloat16`，数值稳定性通常好于 float16。
- `use_flash_attention` 依赖 PyTorch 2.x SDPA，实际是否走 FlashAttention kernel 由 GPU、CUDA、dtype 和输入形状共同决定。
