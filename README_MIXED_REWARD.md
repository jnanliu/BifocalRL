# Mixed-Reward 实验交接说明

本文只说明 `co-learning` 中的单模型 mixed-reward 实验：配置、训练、续跑、评估、结果和画图。项目的完整结构和安装方式见 [README.md](README.md)。

## 1. 环境

常规模型使用 `colearning-self`；Qwen3.5、SmolLM3 等需要新版 Transformers 的模型使用 `colearning-self-qwen35`。同一组对比实验应固定代码提交、conda 环境和 GPU 数量。

```bash
source ~/.zshrc
conda activate colearning-self
cd /home/jizeyu26/jzy/7_co_learning/test/co-learning

git status --short
python -c "import torch, transformers, accelerate, deepspeed; print(torch.__version__, transformers.__version__, accelerate.__version__, deepspeed.__version__)"
nvidia-smi
```

新机器可参考 [环境文档](../docs/CO_LEARNING_ENV_SETUP.md) 和 `requirements.txt`。模型路径及本地数据集根目录必须按机器实际位置修改。

## 2. 核心流程

`scripts/run_single.sh` 按 pipeline step 执行：

```text
读取 JSON
  -> vLLM 为每个 prompt 生成多条回答
  -> 计算 MV / Self-Certainty reward 和 old/ref logps
  -> 构造组内优势及 mixed-reward 权重
  -> GSPO/GRPO 训练
  -> 保存模型及 DeepSpeed optimizer/scheduler state
  -> 运行配置中启用的评估
```

| 位置 | 作用 |
| --- | --- |
| `scripts/run_single.sh` | 总入口、step 循环、断点续跑、checkpoint 清理和评估调度 |
| `src/pipelines/generate_only.py` | 使用 vLLM 生成回答并写入 cache |
| `src/pipelines/score_and_compute_single.py` | 计算 reward、old logps 和 ref logps |
| `src/training/single_intuitor_train.py` | 从 cache 构造训练数据、训练并保存诊断信息 |
| `src/training/co_intuitor_train.py` | GSPO/GRPO loss、优势归一化和 mixed-reward 核心公式 |
| `utils/single_model_trainer.py` | Accelerate/DeepSpeed、optimizer、scheduler 和模型保存 |
| `eval/` | TTRL、LightEval、HACRL 和 ZeroEval 后端 |
| `scripts/plotting/plot_ttrl_eval_auto_new.py` | 汇总各 step 评估结果并画曲线 |

## 3. Baseline 与变体

记 `A_mv` 为 Majority Vote 优势，`A_sc` 为 Self-Certainty 优势。固定混合模式中，`alpha` 是 MV 权重：

```text
A = alpha * A_mv + (1 - alpha) * A_sc
```

| 配置/模式 | 含义 |
| --- | --- |
| `single_intuitor` | 仅使用 Self-Certainty reward |
| `single_mv` | 仅使用 Majority Vote reward |
| `single_repro_gspo` | 使用真值 scorer 的 GSPO 复现基线 |
| `alpha025/050/075` | 固定 `alpha=0.25/0.50/0.75` |
| `adaptive` | 按两个优势的绝对值动态计算 alpha |
| `adaptive_sc` | 按组内 Self-Certainty 动态计算 alpha |
| `adaptive_sc_zscore` | 对 batch 内 Self-Certainty 做映射后得到 alpha |
| `adaptive_sc_zscore_corr` | 在上一模式上叠加 legacy correlation sample weight |
| `reliability_*_all` | Reliability 同时控制优势混合权重和 prompt sample weight |
| `reliability_*_advantage_only` | Reliability 只控制两个优势的混合权重 |
| `reliability_*_sample_only` | 固定 `alpha=0.5`，Reliability 只控制 prompt sample weight |
| `reliability_baseline` | 两种 Reliability 权重都关闭，数值等价于固定 `alpha=0.5` |
| `prism` | `A=A_mv+A_sc`，不做最终 batch z-score normalization |
| `ettrl` | `A=(R_sc / group_mean(R_sc)) * A_mv`，不做最终 batch normalization |

Reliability 目前有 `0.6 SC + 0.4 Corr` 和 `0.4 SC + 0.6 Corr` 两套系数。`all`、`advantage_only`、`sample_only` 各两套，加一个关闭双权重的 baseline，共七个变体。

`reward.mixed_reward_final_normalize` 控制混合优势最后是否在整个 batch 上做 z-score。PRISM 和 eTTRL 必须为 `false`；其余模式以配置值或默认值为准。

## 4. 配置文件

配置位于：

```text
recipes/configs/mixed_reward/single/<ModelFolder>/*.json
```

新增实验时，优先复制同模型、同参数规模下最接近的配置，再修改以下字段：

| 字段 | 说明 |
| --- | --- |
| `name` | 实验唯一名称，也是输出目录名；不要让不同实验共用同一名称 |
| `models.policy` | 初始模型的本地路径 |
| `models.model_tag` | 模型分组标识；绘图脚本依赖该值 |
| `dataset` | 训练数据、prompt 模式、每步 prompt 数和采样方式 |
| `generation` | 每题生成数、温度、prompt/completion 长度 |
| `scoring` | Self-Certainty 和 logps 的推理 batch size |
| `reward.mode` | `intuitor`、`mv`、`truth` 或 `mixed_reward` |
| `reward.beta` | GSPO 中相对 reference policy 的 KL 系数；`0` 表示不加该项 |
| `reward.mixed_reward_*` | alpha、Reliability、相关性权重和最终归一化参数 |
| `training.start_step/max_steps` | 本次 step 范围；新实验通常从 `1` 开始，step 0 是初始模型评估 |
| `training.learning_rate/lr_scheduler` | 学习率和 scheduler；应在开跑前确定完整训练时长 |
| `training.accelerator_config` | DeepSpeed 配置；显存紧张的 3B/4B 模型通常使用 offload 版本 |
| `eval.<backend>.enabled/tasks` | 每步启用的评估后端和数据集 |
| `output.*` | checkpoint、评估结果和保留策略 |

续跑相关参数默认值由 `run_single.sh` 提供：

```json
{
  "training": {
    "persist_optimizer_state": true,
    "require_optimizer_state": true
  },
  "output": {
    "keep_all_checkpoints": false,
    "checkpoint_keep_limit": 2
  }
}
```

- 精确续跑要求上一步的模型 checkpoint、DeepSpeed optimizer/scheduler state 和 `runtime_manifest.json` 完整对应。
- `require_optimizer_state=false` 只用于把旧权重当作 warm start；此时 optimizer 会重建，不能视为严格连续实验。
- `keep_all_checkpoints=false` 会在新 step 完成并评估后只保留最近若干个模型/state pair。已有评估结果不会删除，但被删掉的历史权重不能再补评估。
- 若计划训练后统一补评估全部 step，需要提前设置 `keep_all_checkpoints=true`；否则应在正常训练的 `eval` 配置中逐 step 完成评估。
- 不建议训练中途改变原定 `max_steps` 后再直接比较学习率曲线；cosine scheduler 的计划与总训练更新数有关。

## 5. 运行与续跑

运行一个实验：

```bash
source ~/.zshrc
conda activate colearning-self
cd /home/jizeyu26/jzy/7_co_learning/test/co-learning

bash scripts/run_single.sh \
  recipes/configs/mixed_reward/single/Qwen3-4B-Base/single_mixed_reward_reliability_04sc_06corr_all.json
```

脚本会查找最新完整 checkpoint。若配置仍覆盖后续 step，会自动从下一步继续；重复执行同一命令不需要手工指定模型路径。日志开头应重点检查 `policy`、`steps`、`reward_mode`、`mixed_mode`、`optim_state`、`keep_ckpts` 和有效 batch size。

只验证参数和 shell 逻辑可先执行：

```bash
bash -n scripts/run_single.sh
bash scripts/run_single.sh --help
```

## 6. 评估与结果

常规训练会在每个 step 后运行 JSON 中 `enabled=true` 的后端。常用数学任务 `math_500,gsm8k` 由 TTRL 评估；新增数学任务也可加入 TTRL；LiveCodeBench 使用 LightEval；CRUXEval 使用 HACRL。

对仍保留的历史 checkpoint 补评估：

```bash
bash scripts/run_single.sh \
  recipes/configs/mixed_reward/single/<ModelFolder>/<training-config>.json \
  --eval-only \
  --eval-profile recipes/configs/mixed_reward/single/Qwen2.5-3B/single_mixed_reward_reliability_04sc_06corr_all_extended_eval.json \
  --eval-start-step 0 \
  --eval-end-step 15
```

先检查而不启动评估可追加 `--eval-dry-run`。`profile.strict=true` 时，任一 checkpoint 缺失都会报错；step 0 使用 `models.policy`。

主要输出目录：

```text
# 模型 checkpoint
outputs/mixed_reward/train/<model_tag>/<name>/latest_checkpoint/model/checkpoint-N/

# DeepSpeed optimizer/scheduler state
outputs/mixed_reward/train/<model_tag>/<name>/latest_checkpoint/training_state/checkpoint-N/

# 常规 TTRL 结果
outputs/mixed_reward/eval/<model_tag>/<name>/a_checkpoint-N/comparison_summary.json

# eval-only 结果
outputs/mixed_reward/eval/<model_tag>/<name>-extended-eval/

# Reliability / PRISM / eTTRL 中间量
outputs/mixed_reward/train/<model_tag>/<name>/latest_checkpoint/debug_data/mixed_reward_*_stepN.json
```

诊断 JSON 包含 batch 级 Reliability/权重列表、统计量、排序，以及每个 prompt 下各回答的 MV/Self-Certainty reward、优势和最终权重。

## 7. 画图

编辑 `scripts/plotting/plot_ttrl_eval_auto_new.py` 底部的：

- `model_tags`
- `selected_methods`
- `datasets`
- `start_step` / `end_step`
- `include_step0_baseline`

新增方法时同时向文件顶部的 `METHOD_LABELS` 加入 `name` 和显示名。新增数据集时补充 `DATASET_CONFIG` 及对应读取逻辑。

```bash
pyton scripts/plotting/plot_ttrl_eval_auto_new.py

# 可选输出目录
python scripts/plotting/plot_ttrl_eval_auto_new.py --output-dir /path/to/plots
```

默认图片保存到 `scripts/plotting/output/auto_new/<model_tag>/`。脚本按“常规结果 -> `dataset_method_suffixes` 指定目录 -> `-extended-eval`”查找，因此通常不需要为补跑结果单独改读取代码。`Y_AXIS_CONFIG["limits"] = None` 时纵轴自动缩放。

## 8. 新增变体检查清单

1. 复制同模型下最接近的 JSON，修改唯一 `name` 和 reward 参数。
2. 确认模型路径、`model_tag`、step、学习率、`beta`、batch size 和 offload 设置。
3. 新公式在 `compute_mixed_reward_advantages()` 中实现，并加入参数/shape/NaN 校验。
4. 在 `single_intuitor_train.py` 记录中间 reward、优势和权重。
5. 增加单元测试，确保旧模式数值与运算顺序没有变化。
6. 在绘图脚本 `METHOD_LABELS` 中注册实验 `name`。
7. 先做 JSON 解析和无卡测试，再做小样本 GPU smoke test，最后启动正式实验。

## 9. ACT服务器上跑实验
- CPU：16c 内存：64/128G GPU：2*A100
- 可参考 mix-qwen2.5-1.5b-step30-0.5 自定义作业模板
- 参考命令：
```bash
source ~/.zshrc
conda activate colearning-self
cd /home/jizeyu26/jzy/7_co_learning/test/co-learning
git checkout ACT_new

bash scripts/run_single.sh recipes/configs/mixed_reward/single/xxx/xxx.json
...
```