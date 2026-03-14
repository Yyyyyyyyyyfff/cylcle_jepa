# Cycle-JEPA 实现与后训练指南

## 当前训练状态

| 任务 | GPU | 状态 |
|------|-----|------|
| Phase 1 训练 | GPU 0 | 进行中 |
| Baseline 探针 | GPU 1 | 进行中 |

## 已完成的 Checkpoint

| 模型 | 路径 | 说明 |
|------|------|------|
| Baseline | `ssv2_no_cycle_baseline/latest.pt` | 无 Cycle Loss，已完成 |
| CycleJEPA (单阶段) | `ssv2_cycle_jepa/latest.pt` | 单阶段训练，已完成 |
| Phase 1 | - | 进行中 |
| Phase 2 | - | 待训练 |

---

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: 预训练 Backward Predictor                         │
│  - 数据: ssv2_train.csv (81,901 视频，无标注)              │
│  - 训练: 只训练 backward_predictor，冻结 encoder/predictor  │
│  - 输出: ssv2_training_v2/ssv2_phase1/phase1_best.pt       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: 联合训练                                          │
│  - 数据: ssv2_phase2_train.csv (51,000 视频，无标注)       │
│  - 训练: encoder+predictor+backward_predictor 联合训练     │
│  - 输出: 完整 CycleJEPA checkpoint                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  下游任务: 探针评估                                          │
│  - 数据: ssv2_val_clean.csv (24,777 视频，有标注)          │
│  - 方法: 线性探针 / Few-shot / 检索                        │
│  - 输出: 验证集准确率                                       │
└─────────────────────────────────────────────────────────────┘
```

**无监督训练学到了什么？**

虽然训练时忽略标签（"打开箱子"、"推动物体"等动作描述），但模型通过**预测视频的时空结构**学到了：

1. **物理常识**：物体不会凭空消失、连续运动、重力/碰撞等
2. **动作表征**：起始/进行/结束、持续时间、因果关系
3. **视觉特征**：空间特征（外观、形状）+ 时间特征（轨迹、变化）

探针评估就是检验这些学到的特征质量如何。

---

## 二、当前配置

### 训练配置

文件：`/home/miaochengyu/VJEPA/vjepa2/ssv2_cycle_jepa/params-pretrain.yaml`

```yaml
app: vjepa
folder: /home/miaochengyu/VJEPA/vjepa2/ssv2_cycle_jepa
data:
  batch_size: 32
  crop_size: 256
  dataset_fpcs: [16]
  dataset_type: VideoDataset
  datasets:
    - /home/miaochengyu/VJEPA/vjepa2/ssv2_train.csv
  fps: 4
  num_workers: 4
  patch_size: 16
  tubelet_size: 2
model:
  model_name: vit_giant_xformers
  pred_depth: 12
  pred_embed_dim: 384
  pred_num_heads: 12
  use_rope: true
  use_cycle_loss: true
  cycle_alpha: 0.65
  cycle_pred_depth: 2
  backward_lr_scale: 0.1
meta:
  dtype: bfloat16
  load_checkpoint: true
  pretrain_checkpoint: /home/miaochengyu/VJEPA/vjepa2/vitg.pt
  load_encoder: true
  load_predictor: true
optimization:
  epochs: 3
  ipe: 5280
  lr: 0.0001
  start_lr: 0.00001
  warmup: 1
  weight_decay: 0.04
```

### 基线训练（无Cycle）

用于对比实验：同样配置但禁用 cycle loss 和 backward predictor。

- **配置文件**: `/home/miaochengyu/VJEPA/vjepa2/ssv2_no_cycle_baseline/params-pretrain.yaml`
- **配置差异**:
  - `use_cycle_loss: false`（禁用）
  - 删除了 `backward_lr_scale`, `cycle_alpha`, `cycle_pred_depth`
  - 其他参数完全相同

- **启动命令**:
```bash
cd /home/miaochengyu/VJEPA/vjepa2 && source .venv/bin/activate && nohup python ssv2_no_cycle_baseline/main.py --fname ssv2_no_cycle_baseline/params-pretrain.yaml --debugmode True > ssv2_no_cycle_baseline/train.log 2>&1 &
```

- **日志查看**:
```bash
tail -n 2000 /home/miaochengyu/VJEPA/vjepa2/ssv2_no_cycle_baseline/train.log

tail -f /home/miaochengyu/VJEPA/vjepa2/ssv2_no_cycle_baseline/train.log
```

- **当前状态**: 正在 GPU 1 上训练（与 GPU 0 的 CycleJEPA 同时运行）

### 评估配置

文件：`/home/miaochengyu/VJEPA/vjepa2/ssv2_eval/params-eval.yaml`

用于训练完成后做线性探针评估（SSv2 动作分类准确率）。

---

## 三、数据集

| 数据集 | 路径 | 数量 | 用途 |
|--------|------|------|------|
| SSv2 Phase 1 训练集 | `/home/miaochengyu/VJEPA/vjepa2/ssv2_train.csv` | 81,901 | Phase 1 预训练（无标注） |
| SSv2 Phase 2 训练集 | `/home/miaochengyu/VJEPA/vjepa2/ssv2_phase2_train.csv` | 51,000 | Phase 2 预训练（无标注） |
| SSv2 有标签训练集 | `/home/miaochengyu/VJEPA/vjepa2/ssv2_train_labeled.csv` | 67,215 | 探针评估（无重叠） |
| SSv2 有标签验证集 | `/home/miaochengyu/VJEPA/vjepa2/ssv2_val_clean.csv` | 24,777 | 探针评估（无重叠） |

**注意**：探针评估需要使用无重叠的数据集（ssv2_train_labeled.csv + ssv2_val_clean.csv），避免数据泄露。

**原始视频位置**：
- Phase 1: `/home/miaochengyu/VJEPA/vjepa2/ssv2_part1/` (8万视频)
- Phase 2: `/mnt/ssv2/20bn-something-something-v2/` (22万视频，挑选5.1万)

**训练流程**：
```
Phase 1 (8万无标注) → Phase1 checkpoint
        ↓
Phase 2 (5万无标注) → 完整 CycleJEPA checkpoint
        ↓
下游任务 (有标签验证集) → 探针评估
```

---

## 四、常用命令

### 1. 启动训练（后台运行，SSH 断开后继续）

```bash
cd /home/miaochengyu/VJEPA/vjepa2 && source .venv/bin/activate && nohup python -m app.main --fname ssv2_cycle_jepa/params-pretrain.yaml --debugmode True --devices cuda:0 > train.log 2>&1 &
```

### 2. 查看训练日志（实时更新）

```bash
# 查看最后 20 行
tail -n 2000 /home/miaochengyu/VJEPA/vjepa2/train.log

# 实时跟踪日志
tail -f /home/miaochengyu/VJEPA/vjepa2/train.log
```

按 `Ctrl + C` 退出 tail -f

### 3. 查看 GPU 使用情况

```bash
nvidia-smi


watch -n 1 nvidia-smi
```

### 4. 查看训练进程

```bash
ps aux | grep "app.main" | grep -v grep
```

### 5. 停止训练

```bash
# 找到进程 PID
ps aux | grep python | grep -v grep

# 停止进程
kill <PID>
```

### 6. 断点续训

训练会自动检测 latest.pt 并继续，无需特殊命令。

### 7. 运行评估

```bash
cd /home/miaochengyu/VJEPA/vjepa2 && source .venv/bin/activate && python -m evals.main_distributed --fname ssv2_eval/params-eval.yaml --devices cuda:0
```

### 8. 查看 checkpoint 文件

```bash
ls -la /home/miaochengyu/VJEPA/vjepa2/ssv2_cycle_jepa/*.pt
```

### 行为说明

| 情况 | 行为 |
|------|------|
| 第一次训练 | 从头开始，加载 vitg.pt 预训练权重 |
| 中途断电/断网 | 重新运行相同命令，自动从 latest.pt 继续 |
| 手动停止 (Ctrl+C) | 自动保存 checkpoint 到 latest.pt |


---

## 五、评估命令

训练完成后，运行线性探针评估：

### Baseline 探针训练（无 Cycle）

```bash
cd /home/miaochengyu/VJEPA/vjepa2 && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=1 nohup python -m evals.main --fname ssv2_no_cycle_baseline_eval/params-eval.yaml --devices cuda:1 --debugmode True > ssv2_no_cycle_baseline_eval/train.log 2>&1 &
```

日志查看：
```bash
tail -f /home/miaochengyu/VJEPA/vjepa2/ssv2_no_cycle_baseline_eval/train.log
```

查看最终结果：
```bash
tail -5 /home/miaochengyu/VJEPA/vjepa2/ssv2_no_cycle_baseline_eval/train.log
```

**参数信息**：
- 训练集：67,215 视频（ssv2_train_labeled.csv）
- 验证集：24,777 视频（ssv2_val_clean.csv）
- Batch Size: 32
- Iterations: 约 2,100

### CycleJEPA 探针训练

```bash
cd /home/miaochengyu/VJEPA/vjepa2 && source .venv/bin/activate && python -m evals.main_distributed --fname ssv2_eval/params-eval.yaml --devices cuda:0
```

### 评估指标

- **验证集准确率**（最重要的指标）

### 准确率参考

| 准确率 | 评价 |
|--------|------|
| < 60% | 较差 |
| 60-65% | 一般 |
| 65-70% | 良好 |
| 70-75% | 优秀 |
| > 75% | 极佳（接近原始 V-JEPA2） |

---

## 六、日志说明

训练时会显示 loss_forward 和 loss_cycle：

```
[1,   100] loss: 0.751 (f: 0.520, c: 0.231) masks: [16: 1.0] [wd: 4.00e-02] [lr: 1.03e-05] [mem: 3.43e+04] [iter: 6773.8 ms] [gpu: 6690.0 ms] [data: 50.9 ms]
```

| 字段 | 含义 |
|------|------|
| loss | 总损失 = 0.65 × f + 0.35 × c |
| f | loss_forward (前向预测损失) |
| c | loss_cycle (循环重建损失) |
| lr | 学习率 |
| mem | 显存使用 (MB) |
| iter | 迭代时间 (ms) |
| gpu | GPU 计算时间 (ms) |
| data | 数据加载时间 (ms) |

---

## 七、修改记录

### 已完成的代码修改

1. **新增 BackwardPredictor** (`src/models/predictor.py`)
   - 循环重建的预测器

2. **支持断点续训** (`app/vjepa/train.py`)
   - load_checkpoint: true
   - 每 1000 iter 保存一次

3. **分离 loss 日志** (`app/vjepa/train.py`)
   - 同时显示 loss_forward 和 loss_cycle

---

## 八、给新 AI 的指引

### 如果要继续训练

```bash
cd /home/miaochengyu/VJEPA/vjepa2 && source .venv/bin/activate && python -m app.main --fname ssv2_cycle_jepa/params-pretrain.yaml --debugmode True --devices cuda:0
```

### 如果要做评估

```bash
cd /home/miaochengyu/VJEPA/vjepa2 && source .venv/bin/activate && python -m evals.main_distributed --fname ssv2_eval/params-eval.yaml --devices cuda:0
```

### 关键文件

| 文件 | 说明 |
|------|------|
| `ssv2_cycle_jepa/params-pretrain.yaml` | 训练配置 |
| `ssv2_eval/params-eval.yaml` | 评估配置 |
| `ssv2_train.csv` | 无标注训练集 |
| `ssv2_phase2_train.csv` | Phase 2 训练集 |
| `ssv2_train_labeled.csv` | 有标签训练集（探针用） |
| `ssv2_val_clean.csv` | 有标签验证集（探针用，无重叠） |
| `vitg.pt` | 预训练权重 |
| `app/vjepa/train.py` | 训练代码 |
| `src/models/predictor.py` | 模型定义 |

### 注意事项

1. **tubelet_size=2** 必须与预训练一致
2. **dataset_fpcs=16** 必须与预训练一致
3. **load_checkpoint=true** 用于断点续训
4. **save_every_freq=1000** 每 1000 iter 保存一次

---

*最后更新：2026-03-12 16:30*

---
2026.3.13更新
## 你需要理解：九、Checkpoint 最优保存 - 修改计划

### 1. 问题说明

当前 checkpoint 保存逻辑：
- 只保存 `latest.pt`（每个 epoch 覆盖）
- 保存的是"最新"参数，不是"最优"参数
- 如果 loss 反弹，最终保存的是"坏" checkpoint

---

### 2. 涉及文件

| 序号 | 文件路径 |
|------|----------|
| 1 | `/home/miaochengyu/VJEPA/vjepa2/app/vjepa/train.py` |
| 2 | `/home/miaochengyu/VJEPA/vjepa2/ssv2_no_cycle_baseline/app/vjepa/train.py` |

---

### 3. 修改内容

#### 修改1：初始化 best_loss 变量

**位置**：约 line 423（`# -- TRAINING LOOP` 之后，`for epoch in range(start_epoch, num_epochs):` 之前）

**原代码**：
```python
# -- TRAINING LOOP
for epoch in range(start_epoch, num_epochs):
```

**修改后**：
```python
# -- TRAINING LOOP
best_loss = float('inf')
best_checkpoint_path = os.path.join(folder, "best.pt")

for epoch in range(start_epoch, num_epochs):
```

---

#### 修改2：每 N 个 iteration 保存最优 checkpoint

**位置**：约 line 716（`log_stats()` 之后，`assert not np.isnan(loss)` 之前）

**原代码**：
```python
log_stats()
assert not np.isnan(loss), "loss is nan"
```

**修改后**：
```python
log_stats()

# 每100个iteration判断一次，保存最优checkpoint
if itr % 100 == 0:
    if loss < best_loss:
        best_loss = loss
        save_checkpoint(epoch * ipe + itr, best_checkpoint_path)
        logger.info(f"[BEST] itr {itr}: loss {loss:.5f} < best_loss {best_loss:.5f}, saved best checkpoint")

assert not np.isnan(loss), "loss is nan"
```

---

#### 修改3：每个 epoch 结束后也保存最优（可选保险）

**位置**：约 line 723（`save_checkpoint(epoch + 1, latest_path)` 之后）

**原代码**：
```python
if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
    save_checkpoint(epoch + 1, latest_path)
    if save_every_freq > 0 and epoch % save_every_freq == 0:
```

**修改后**：
```python
if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
    save_checkpoint(epoch + 1, latest_path)
    
    # epoch结束后也保存最优（保险）
    if loss_meter.avg < best_loss:
        best_loss = loss_meter.avg
        save_checkpoint(epoch + 1, best_checkpoint_path)
    
    if save_every_freq > 0 and epoch % save_every_freq == 0:
```

---

### 4. 保存效果

| 文件名 | 触发条件 | 内容 |
|--------|----------|------|
| `latest.pt` | 每个 epoch 结束 | 最后一个 epoch 的参数 |
| `best.pt` | loss 低于历史最佳时覆盖 | 历史最低 loss 对应的参数 |

---

### 5. 注意事项

1. **itr % 100** - 可根据需要调整为 50、200、500 等
2. **两个文件都需要修改** - 保持代码一致
3. **不影响训练** - 保存逻辑独立，不干扰梯度更新


以下是辅助你思考checkpoint的内容：仔细检查所有相关代码。我们先修改checkpoint的最优保存。你先给我制定一个详细的修改计划，要求遵循SOLID原则和第一性原理。
---

## 十、两阶段训练策略

### 1. 训练策略说明

采用两阶段训练，让 backward_predictor 先学习如何重建，再"倒逼" encoder 和 forward predictor 学习更完整的表征。

### 2. 训练流程

| 阶段 | Encoder | Predictor | Backward Predictor | 数据集 | 损失函数 |
|------|---------|-----------|-------------------|--------|----------|
| **Phase 1** | 冻结（不更新） | 冻结（不更新） | 用 loss2 更新 | ssv2 训练集 | loss_cycle |
| **Phase 2** | 用 combined loss 更新 | 用 combined loss 更新 | 用 loss2 单独更新（继承 Phase 1 权重） | 新的 ssv 数据 | loss_forward + loss_cycle |

#### Phase 1（阶段1）：
- 加载预训练权重的 encoder 和 predictor
- **冻结** encoder 和 predictor（设置 `requires_grad = False`，完全不更新）
- 在 ssv2 训练集上只训练 backward_predictor
- backward_predictor 通过 loss2 (loss_cycle) 学习重建

#### Phase 2（阶段2）：
- **解除** encoder 和 predictor 的冻结
- 在新的 ssv 数据上训练
- Encoder + predictor 用 **combined loss** (loss1 + loss2) 更新
- backward_predictor 用 **loss2** 单独更新
- backward_predictor 的**初始权重继承自 Phase 1**，并且在 Phase 2 继续更新

---

### 3. 文件结构

| 文件 | 职责 | 说明 |
|------|------|------|
| `app/vjepa/train_phase1.py` | Phase 1 | 冻结 encoder/predictor，只训练 backward_predictor |
| `app/vjepa/train.py` | Phase 2 | 解除冻结，使用 combined loss 训练（已有逻辑） |

---

### 4. train_phase1.py 实现要点

#### 4.1 加载预训练权重后冻结 encoder 和 predictor

```python
# 加载预训练权重
encoder, predictor, target_encoder, backward_predictor = load_pretrained(...)

# 冻结 encoder 和 predictor
for p in encoder.parameters():
    p.requires_grad = False
for p in predictor.parameters():
    p.requires_grad = False
```

#### 4.2 只训练 backward_predictor

在 train_step 中，只对 backward_predictor 进行反向传播和参数更新：

```python
# 只 backward backward_predictor (loss_cycle)
loss_cycle.backward()

# 只更新 backward_predictor
optimizer_backward.step()
optimizer_backward.zero_grad()

# encoder 和 predictor 不更新（保持冻结）
```

#### 4.3 保存 checkpoint

Phase 1 训练完成后保存 checkpoint，包含：
- encoder 权重（冻结状态）
- predictor 权重（冻结状态）
- backward_predictor 权重（训练好的）
- optimizer_backward 状态

---

### 5. train.py (Phase 2) 已有逻辑

当前 `app/vjepa/train.py` 已实现 Phase 2 的训练逻辑：

```python
# Phase 2: encoder+predictor 用 combined loss，backward_predictor 用 loss2 单独更新

# 1. 计算 loss
loss_forward = loss_fn(z, h)      # loss1
loss_cycle = cycle_loss_fn(s_x, h)  # loss2
loss = cycle_alpha * loss_forward + (1 - cycle_alpha) * loss_cycle

# 2. backward_predictor 单独更新 (loss2)
scaler_backward.scale(loss_cycle).backward(retain_graph=True)
scaler_backward.step(optimizer_backward)
scaler_backward.zero_grad()

# 3. encoder + predictor 用 combined loss 更新
scaler_main.scale(loss).backward()
scaler_main.step(optimizer_main)
scaler_main.zero_grad()
```

---

### 6. 配置参数

建议新增参数到 `params-pretrain.yaml`：

```yaml
# 训练阶段控制
training_phase: 2  # 1 或 2
phase1_epochs: 5  # Phase 1 训练多少个 epoch

# Phase 2 新数据集路径
phase2_data:
  datasets:
    - /path/to/new_ssv_data.csv  # 待填写
```

---

### 7. 实施状态

- [x] Phase 1 训练配置：ssv2_training_v2/ssv2_phase1/
- [x] Phase 2 训练配置：ssv2_training_v2/ssv2_phase2/
- [x] Phase 2 数据集：ssv2_phase2_train.csv (51000个视频)

---

## 十一、两阶段训练启动命令

注意注意！！checkpoint和这个两阶段训练的代码文件，请你不要在原有文件基础上进行修改。而是创建新的目录，在新目录里面创建新的文件去写。相同的文件复制粘贴一份即可。总而言之：不要修改我目前有的文件。

---

## 十一、两阶段训练启动命令

### 目录结构

```
/home/miaochengyu/VJEPA/vjepa2/ssv2_training_v2/
├── ssv2_phase1/        # Phase 1 训练
│   ├── main.py
│   ├── params-pretrain.yaml
│   └── app/vjepa/train.py
└── ssv2_phase2/        # Phase 2 训练
    ├── main.py
    ├── params-pretrain.yaml
    └── app/vjepa/train.py
```

### Phase 1 启动命令

```bash
cd /home/miaochengyu/VJEPA/vjepa2 && source .venv/bin/activate

nohup python -m app.main --fname ssv2_training_v2/ssv2_phase1/params-pretrain.yaml --debugmode True --devices cuda:0 > ssv2_training_v2/ssv2_phase1/train.log 2>&1 &
```

### Phase 2 启动命令（等 Phase 1 完成）

```bash
cd /home/miaochengyu/VJEPA/vjepa2 && source .venv/bin/activate

nohup python -m app.main --fname ssv2_training_v2/ssv2_phase2/params-pretrain.yaml --debugmode True --devices cuda:0 > ssv2_training_v2/ssv2_phase2/train.log 2>&1 &
```

### 日志查看

```bash
# Phase 1
tail -f /home/miaochengyu/VJEPA/vjepa2/ssv2_training_v2/ssv2_phase1/train.log

# Phase 2
tail -f /home/miaochengyu/VJEPA/vjepa2/ssv2_training_v2/ssv2_phase2/train.log
```

---

## 十二、下游任务评估

### 任务1：174类线性探针（对比实验）

评估 Baseline vs CycleJEPA 的基础特征质量。

```bash
# Baseline 探针（已在 GPU 1 运行）
cd /home/miaochengyu/VJEPA/vjepa2 && source .venv/bin/activate
nohup python -m evals.main --fname ssv2_no_cycle_baseline_eval/params-eval.yaml --devices cuda:1 --debugmode True > ssv2_no_cycle_baseline_eval/train.log 2>&1 &

# CycleJEPA 探针
nohup python -m evals.main --fname ssv2_training_v2/ssv2_cycle_jeapa_eval/params-eval.yaml --devices cuda:1 --debugmode True > ssv2_training_v2/ssv2_cycle_jeapa_eval/train.log 2>&1 &
```

### 任务2：Few-shot 分类

评估模型的泛化能力（5-way 5-shot）。

```bash
cd /home/miaochengyu/VJEPA/vjepa2 && source .venv/bin/activate

# Baseline
nohup python evals/few_shot_classification/eval.py --fname ssv2_training_v2/ssv2_few_shot_eval/params-eval.yaml > ssv2_training_v2/ssv2_few_shot_eval/baseline.log 2>&1 &
```

配置文件：`ssv2_training_v2/ssv2_few_shot_eval/params-eval.yaml`

### 任务3：视频检索

评估特征相似度区分能力。

```bash
cd /home/miaochengyu/VJEPA/vjepa2 && source .venv/bin/activate

# CycleJEPA
nohup python evals/video_retrieval/eval.py --fname ssv2_training_v2/ssv2_retrieval_eval/params-eval.yaml > ssv2_training_v2/ssv2_retrieval_eval/cycle_jeapa.log 2>&1 &

# Baseline
# 修改 checkpoint 路径后运行
```

### 下游任务文件结构

```
/home/miaochengyu/VJEPA/vjepa2/
├── evals/
│   ├── video_classification_frozen/   # 任务1：174类探针
│   ├── few_shot_classification/        # 任务2：Few-shot
│   └── video_retrieval/               # 任务3：检索
└── ssv2_training_v2/
    ├── ssv2_cycle_jeapa_eval/         # 任务1配置
    ├── ssv2_few_shot_eval/            # 任务2配置
    └── ssv2_retrieval_eval/           # 任务3配置
```

---

*最后更新：2026-03-14 14:45*
