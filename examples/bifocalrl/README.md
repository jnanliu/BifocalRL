# BifocalRL / 无监督 RL baselines on OpenRLHF

独立迁移到 `jnanliu/OpenRLHF`，训练后端为 **DeepSpeed + vLLM + Ray**。
运行时只需这个 `openrlhf/` checkout，不导入上层 BifocalRL 或旁边的 slime。
原 BifocalRL 和 slime 的代码保持原状。

## 方法

| recipe | 行为 |
| --- | --- |
| `ttrl` | 问题内 majority vote 产生二元伪奖励，group 标准化 |
| `intuitor` | 当前 actor 完整词表 SCE，group 标准化 |
| `mixed` | 原 MV/SCE 分别标准化、融合、最终 batch z-score |
| `reliability_sc_corr` | 原 reliability 融合及样本权重 |
| `legacy_prism` | 原仓库旧公式 `A_MV + A_SCE` |
| `legacy_ettrl` | 原仓库旧公式 `SCE / mean_group(SCE) * A_MV` |
| `ettrl` | ETMR 树状 rollout + AdvClip + AdvRes |
| `ettrl_etmr` | 只启用 ETMR |
| `ettrl_adv_clip` | 并行 rollout，EAR AdvClip，阈值 2 |
| `ettrl_adv_res` | 并行 rollout，EAR AdvRes，相对调整截断到 ±0.2 |
| `covo` | 论文公式的距离矩阵、Con/Vol、簇内向量奖励与 curiosity |
| `covo_released` | CoVo 作者公开代码的奖励约定 |
| `prism` | 独立 GenPRM 的 process/completion 奖励与 SCE 优势融合 |
| `prism_quadratic` | PRISM，SCE 权重做二次衰减 |

新增论文：[ETTRL](https://arxiv.org/abs/2508.11356)、
[CoVo](https://arxiv.org/abs/2506.08745)、[PRISM](https://arxiv.org/abs/2601.04700)。
`legacy_*` 保留旧命名的数学行为，新 `ettrl` / `prism` 对应论文方法。
这些是方法迁移与示例配置，尚未通过训练复现论文分数。

## 安装与启动

按本 checkout 的 [OpenRLHF 安装说明](../../README.md) 准备完整 GPU 环境。
依赖版本应以该 fork 的 `requirements.txt` 为准，包含 DeepSpeed、vLLM、Ray 和
Transformers；不要覆盖旧 BifocalRL 的 Python 环境。无需转换成 Megatron checkpoint。

```bash
cd openrlhf
export HF_CHECKPOINT=/path/to/actor_hf_checkpoint
export PROMPT_DATA=/path/to/train.jsonl
export NUM_GPUS=8
bash examples/bifocalrl/run.sh mixed
```

把 `mixed` 替换成上表任意 recipe。底层直接使用原生入口：

```bash
python -m openrlhf.cli.train_ppo_ray \
  --bifocal.config examples/bifocalrl/configs/ttrl.json \
  ...其他 OpenRLHF 参数...
```

该 fork 使用点分层级参数，如 `--actor.model_name_or_path`、
`--data.prompt_dataset`、`--rollout.n_samples_per_prompt`，
不应混用其他 OpenRLHF 版本的旧 CLI 名称。
未提供 `--bifocal.config` 或 `--bifocal.method` 时继续使用原生算法路径。

## 数据与评估

统一为 `prompt` 列的 JSONL，可从 HF 或本地数据转换：

```bash
python examples/bifocalrl/prepare_data.py DigitalLearningGmbH/MATH-lighteval data/train.jsonl \
  --split train --prompt-key problem
python examples/bifocalrl/prepare_data.py HuggingFaceH4/MATH-500 data/math500.jsonl \
  --split test --prompt-key problem --label-key answer
python examples/bifocalrl/prepare_data.py openai/gsm8k data/gsm8k.jsonl \
  --subset main --split test --prompt-key question --label-key answer --label-format gsm8k
```

默认输出：

```json
{"prompt": [{"role": "user", "content": "问题，请把最终答案放入 \\boxed{}"}]}
```

训练数据无需 label。投票保留原仓库 boxed 数学答案抽取逻辑，prompt 应明确要求
`\boxed{...}`。`--plain-prompts` 保留字符串格式，可配合
`APPLY_CHAT_TEMPLATE=0` 用于没有 chat template 的 base 模型。
`--revision` 可固定 HF 数据版本，`--limit` 用于小规模检查。

OpenRLHF 原生 `--data.prompt_dataset` 也支持 HF repo ID、JSON/JSONL、Parquet
及 `save_to_disk` 目录；直接使用 HF 数据时按实际列名设置 `--data.input_key`。
需要 subset/config 的数据可先通过转换脚本处理。

启用独立有标签评估：

```bash
EVAL_DATA=data/math500.jsonl EVAL_STEPS=10 \
  bash examples/bifocalrl/run.sh reliability_sc_corr
```

评估默认从 `label` 列读取正确答案，可在算法 JSON 中设置 `eval_label_key`。
这一覆盖仅作用于 eval Dataset，训练集仍可完全没有标签列。
评估走普通并行生成与数学答案判定，不运行 ETMR、PRM 或训练伪奖励。
原生 eval 记录 datasource 的 pass@1/pass@k，可在数据中加入 `datasource` 区分 benchmark。
多个评估文件可用 `EVAL_DATA=file1.jsonl,file2.jsonl`；此时文件需使用相同 prompt/label schema。
本目录未内置或下载 benchmark 数据。

## 配置与环境变量

专用 CLI：`--bifocal.config`、`--bifocal.method`、`--bifocal.prm_url`、
`--bifocal.prm_model`。JSON 默认值见
[`runtime.py`](../../openrlhf/algorithms/bifocalrl/runtime.py)。
融合接受原 `MixedRewardConfig` 的全部字段，例如：

```json
{
  "method": "mixed",
  "mixed": {
    "alpha": 0.5,
    "alpha_mode": "reliability_sc_corr",
    "reliability_use_sample_weight": true,
    "reliability_use_advantage_weight": true,
    "final_normalize": true
  }
}
```

保留的融合模式还有 `fixed`、`adaptive`、`adaptive_sc`、`adaptive_sc_zscore`，
以及单独命名的旧 `legacy_prism` / `legacy_ettrl`。

`run.sh` 支持以下环境变量，末尾参数原样透传至 OpenRLHF：

| 变量 | 默认 / 用途 |
| --- | --- |
| `HF_CHECKPOINT`, `PROMPT_DATA` | 必填，actor HF checkpoint 和训练数据 |
| `BIFOCAL_CONFIG` | 默认按第一个脚本参数选择 JSON |
| `OUTPUT_DIR` | `output/bifocalrl-<recipe>`，包含 final/checkpoints |
| `NUM_GPUS`, `VLLM_TP_SIZE` | 8、1；vLLM engine 数为二者之商 |
| `GROUP_SIZE`, `PROMPTS_PER_ROLLOUT` | 8、16 |
| `MICRO_TRAIN_BATCH_SIZE`, `MICRO_ROLLOUT_BATCH_SIZE` | 1、1 |
| `NUM_EPISODES`, `MAX_SAMPLES` | 1、100000000 |
| `MAX_LENGTH`, `MAX_RESPONSE_LEN`, `TEMPERATURE` | 4096、3072、1.0 |
| `LR`, `VLLM_MEMORY_FRACTION` | 1e-6、0.5 |
| `PROMPT_SPLIT`, `APPLY_CHAT_TEMPLATE` | train、1 |
| `EVAL_DATA`, `EVAL_SPLIT`, `EVAL_STEPS` | 未设置、train、10 |
| `PRM_URL`, `PRM_MODEL` | 独立 GenPRM 服务及匹配的 tokenizer |
| `PYTHON` | python |

## ETTRL / CoVo / PRISM 细节

ETMR 的 `GROUP_SIZE = tree_roots*(1+tree_forks*tree_branches)`，示例 `2*(1+3*1)=8`。
按 Algorithm 1 的 sampled surprisal 选 fork，复用 root 的 token/logprob 前缀，
从 fork token 重新采样；完整前缀也纳入 action mask。
EAR 使用 actor 完整词表 Shannon entropy，而不是 sampled logprob 的负数。
`ear_mode` 为 `none/clip/res/clip_res`，`ear_beta` 控制 AdvClip，
`ear_res_clip` 控制相对调整截断，null 为直接公式。
目前所有树叶都参与投票与训练，没有默认执行论文实验的 vote-then-subsample 预算。
ETMR 当前要求 top_p=1；root 少于 `tree_forks` 个 token 时明确报错。

CoVo 通过当前 vLLM engine 的 `prompt_logprobs` 对每个中间 prefix 和候选答案
执行 teacher forcing，不加载滞后的 HF 策略副本。
vLLM 为该请求额外生成一个 token，此 token 被丢弃，不进入奖励或训练。
相关接口见 [vLLM SamplingParams](https://docs.vllm.ai/en/v0.14.0/api/vllm/sampling_params/)。
按换行分步、默认最多合并为 15 步；开销约为 responses × steps × distinct answers。

`covo_variant: paper` 使用论文 Con、Vol、连续向量聚合和 Eq.10。
其中 token 概率分布解释为该步观测 token 的概率在位置维归一化。
`released` 根据[作者代码](https://github.com/sastpg/CoVo)保留严格 Con 比较、
1-based Vol、perplexity/cube-root-KL curiosity、阈值化 intrinsic 和 0.4 curiosity 奖励。
这些是公开代码的奖励约定，不是其所有评分边界行为的逐字移植。
`covo_aggregation` 可为 vector/linear；`covo_curiosity_weight: 0` 可做消融。

PRISM 必须接入真实 PRM。例如在训练 actor 以外的设备启动 SGLang GenPRM：

```bash
# 用实际空闲设备替换这里的 GPU 号，并单独设置训练进程可见的 GPU。
CUDA_VISIBLE_DEVICES=8 python -m sglang.launch_server \
  --model-path GenPRM/GenPRM-7B --port 31000
PRM_URL=http://127.0.0.1:31000 PRM_MODEL=GenPRM/GenPRM-7B \
  bash examples/bifocalrl/run.sh prism
```

整份答案交给 GenPRM，逐段输出 judgement 后给 completion verdict，
再在实际判定上下文 teacher-force `Yes`，读取 P(Yes)，包括生成 No 的情况。
process 概率取 min，与 completion 概率做 harmonic mean；
最终优势为 `gamma*A_SCE + A_PRM`，两个通道各自按问题标准化。
默认提示词依据论文接口编写，并非逐字复刻其附录；`prm_prompt_file` 可覆盖。
格式或服务失败会报错，不以 MV 替代。

也可设置 `prm_function: "your_package.module.score"`：

```python
async def score(config, sample):
    # 通过真实 PRM 推理得到概率，不使用正确答案。
    return {"step_probabilities": step_probabilities,
            "completion_probability": completion_probability}
```

`prism_gamma_schedule` 可为 constant/quadratic。
二次衰减默认按 rollout 批次数估计，`prism_gamma_steps` 可显式固定总批数，
断点续训使用恢复的 global_step。

## 数值、扩展和限制

- SCE 保留原定义 `logsumexp(logits)-mean(logits)`，即 `KL(U||p)+log(V)`。
  SCE/熵在 actor 的 temperature scaling 之前计算，PPO logprob 仍使用原生温度逻辑。
- 原融合 group std 使用 sample std，batch z-score 使用 population std；
  ETTRL 的 GRPO std 按论文使用 population std。
- CoVo 使用 gamma=1、outcome-return Reinforce++，按完整 rollout 内所有有效
  action token 白化，并沿用上一份迁移的 Bessel correction 约定。
- 所有优势先恢复 Experience.index 对应的原始问题顺序，再计算 query/batch 统计，
  避免 dynamic microbatch 的重排改变融合权重。
- 原生日志之外保留 `bifocal_mv/sc/entropy/aux/gamma`；Intuitor 的 reward 日志更新为 SCE。
- 当前支持同步、单策略 actor、纯文本、单轮生成。没有迁移双 actor co-training 调度。
  SCE/熵目前要求 DeepSpeed tensor_parallel_size=1；ZeRO、DP、原生 packing/ring
  attention 的统计量提取路径保留。ring attention 的 GPU 路径尚未实测。
- 禁止额外 critic、外部 outcome RM、长度奖励惩罚和 reward-side KL。
  需要 reference KL 时使用 `--algo.kl.use_loss --algo.kl.init_coef ...`。
- 续训沿用原生 `--ckpt.load_enable`，保持算法 JSON 与原运行一致。

代码入口：`openrlhf/algorithms/bifocalrl/runtime.py`（配置与全局优势）、
`rollout.py`（vLLM 分组、ETMR/CoVo/PRISM）、`methods.py`（纯公式）、
`mixed_reward/`（原融合注册表）。增加融合策略可用原 `register_mixed_reward_mode`。
框架接线位于 CLI、Actor.forward、PolicyModelActor.forward、SamplesGenerator、
vLLM group dispatcher 及 RemoteExperienceMaker。

## 验证

```bash
python -m compileall -q openrlhf examples tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -o addopts='' tests/bifocalrl
bash -n examples/bifocalrl/run.sh
```

测试覆盖原融合输出逐值一致、原生 Actor 的统计量温度与 token 对齐、重排后的全局
标准化、ETMR 前缀与 logprob/action mask、真实 Experience padding/batching、
当前 vLLM 的 CoVo teacher forcing、PRM 概率聚合、eval 标签隔离和 CLI 配置接线。
纯 CPU 测试使用轻量边界替身，不声称启动过实际 vLLM 或 DeepSpeed 模型。

完整上游测试命令已尝试；当前 CPU 环境缺少 DeepSpeed/datasets，并且 Transformers
早于此 fork 所需版本，部分上游模块无法加载。GPU 训练、真实 GenPRM 服务及论文性能
仍需在完整 GPU 环境中验收。
