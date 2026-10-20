# BifocalRL / 无监督 RL baselines on OpenRLHF

独立迁移到 `jnanliu/OpenRLHF`，训练后端为 **DeepSpeed + vLLM + Ray**。
运行时只需这个 `openrlhf/` checkout，不导入上层 BifocalRL 或旁边的 slime。
原 BifocalRL 和 slime 的代码保持原状。

## 方法

| recipe | 行为 |
| --- | --- |
| `ttrl` | 问题内 majority vote 产生二元伪奖励，group 标准化 |
| `intuitor` | 当前 actor 完整词表 SCE，group 标准化 |
| `bifocalrl`（BifocalRL） | MV/SCE 分别标准化、原始信号过滤、保留集内 SC 答案簇分解、等权融合与 query 重加权、最终 batch z-score |
| `reliability_sc_corr` | 独立方法：按一致率与 MV/SCE 相关性计算 query weight 和融合权重 |
| `ettrl` | ETMR 树状 rollout + AdvClip + AdvRes |
| `covo` | 论文公式的距离矩阵、Con/Vol、簇内向量奖励与 curiosity |
| `prism` | 独立 GenPRM 的 process/completion 奖励与 SCE 优势融合 |
| `circa`（CIRCA） | Cluster voting + 格式奖励、SCE 分通道标准化，以及 token 级 KL-Cov 损失 |

新增论文：[ETTRL](https://arxiv.org/abs/2508.11356)、
[CoVo](https://arxiv.org/abs/2506.08745)、[PRISM](https://arxiv.org/abs/2601.04700)、
[Two is better than one](https://arxiv.org/abs/2605.22620)。
只提供上表八个方法和同名 recipe，已移除旧公式、其他融合模式及额外 recipe。
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
bash examples/bifocalrl/run.sh bifocalrl
```

默认 recipe 为 `bifocalrl`，也可替换成上表任意 recipe。底层直接使用原生入口：

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
原 `mixed` 方法现命名为 **BifocalRL**，方法标识和配置文件名使用 `bifocalrl`。
`--bifocal.method BifocalRL` 和 `bash examples/bifocalrl/run.sh BifocalRL` 也可使用。
融合参数块仍使用 `mixed`。`bifocalrl` 使用等权融合与下述过滤流程；
`reliability_sc_corr` 保留原可靠性融合及归一化后的 query weight，配置示例：

```json
{
  "method": "reliability_sc_corr",
  "mixed": {
    "alpha": 0.5,
    "alpha_mode": "reliability_sc_corr",
    "reliability_use_sample_weight": true,
    "reliability_use_advantage_weight": true,
    "final_normalize": true
  }
}
```

`reliability_use_sample_weight` 控制 query weight，
`reliability_use_advantage_weight` 控制 MV/SCE 融合权重，可独立关闭；
这两个开关仅作用于 `reliability_sc_corr`。

## BifocalRL：GD²PO 过滤与答案簇内 SC 分解

运行 `bash examples/bifocalrl/run.sh bifocalrl`。对当前完整 rollout batch 的每道题，
保留 MV 二元奖励和 actor SCE 两个通道，分别按该题的全部回答做 sample-std
group normalization，得到 `A_MV` 与 `A_SC`。先用这两个原始信号过滤，
再在同题保留回答中按最终答案的数学等价类分解 SC：

```text
B_ij      = 0.5 * A_MV_ij + 0.5 * A_SC_ij
SNR_ij    = abs(B_ij) / (0.5*abs(A_MV_ij) + 0.5*abs(A_SC_ij) + eps)
keep_ij   = SNR_ij > 0.5
q_i       = mean_j(keep_ij)
mu_ia     = mean(A_SC_ij for retained j with answer cluster a)
within_ij = A_SC_ij - mu_i,answer(j)
SC_new_ij = within_ij + gamma * mu_i,answer(j)
C_ij      = q_i * (0.5 * A_MV_ij + 0.5 * SC_new_ij)
A_ij      = (C_ij - mean(C[keep])) / (std_population(C[keep]) + eps), if keep_ij
            0, otherwise
```

融合权重始终为 `0.5/0.5`，SNR 阈值默认为固定的 `0.5`，可通过配置调整。
`mixed.sc_between_gamma` 默认为 `0.5`：保留簇内差异，减弱 SC 答案簇均值。
设为 `1` 精确恢复原融合；设为 `0` 则保留样本中每个答案簇的 SC 贡献均值为零。
该公式直接扣除 `(1-gamma)*mu`，没有额外扣除/恢复 query 均值，也不对分量重新标准化。
因此 gamma 也会改变过滤后 query 的 SC 均值和整体幅度。

答案聚类复用现有数学等价判断（如 `1/2` 与 `0.5`），仅在同一道题内进行。
解析失败的回答各自作为独立簇；被过滤回答不计入簇均值和分母。
仅保留一条回答的簇，其簇内残差为零；没有保留回答的簇不参与计算。
分解后不再次过滤，所以保留集不随 gamma 在同一批 rollout 上变化，
但调整后的信号可能不再满足原 SNR 阈值。

同一道题的过滤判断不受其他问题组成影响；最终 batch normalization 的统计量
仍由当前 batch 中保留的回答共同决定。

SNR 及 query 保留比例采用 [GD²PO Eq.11–12、19](https://arxiv.org/html/2606.16771v1#S3)。
按当前实现要求，query 重加权发生在最终 batch normalization 之前，之后不再乘权重。
该归一化覆盖完整 rollout batch 中保留的回答，不受训练 micro-batch 重排影响。
因此重加权影响不同问题之间的相对强度，共同缩放会被最终归一化消去。

被过滤的回答不参与最终均值/方差统计，最终 advantage 严格为 0。
整批全部被过滤、仅保留一条或保留值无差异时，最终 advantage 为全 0。
过滤通过 advantage 屏蔽策略奖励项，保留原分组及原生 loss 分母；reference KL、
entropy 等独立训练项按原配置计算，不执行额外重采样。

相关字段位于 recipe 的 `mixed` 中：

| 字段 | 默认值 | 作用 |
| --- | --- | --- |
| `alpha`, `final_normalize` | `0.5`, `true` | BifocalRL 固定要求等权融合、最终 batch normalization |
| `gd2po_filter_mode` | `snr` | `snr` 为上述过滤，`hard` 为异号过滤，`none` 关闭过滤 |
| `gd2po_snr_threshold` | `0.5` | SNR 基础阈值 |
| `gd2po_query_reweight` | `true` | 按该题保留比例重加权；关闭时使用 1 |
| `sc_between_gamma` | `0.5` | `[0,1]`，保留集合内 SC 答案簇均值的权重；`1` 恢复原融合 |

`hard` 把 0 视为中性，仅屏蔽两个通道严格异号的回答，过滤判断使用归一化后的优势。
关闭 filtering 时，保留比例为 1，在完整 group 上进行 SC 分解；
同时设置 `sc_between_gamma: 1` 才恢复原等权融合加 batch normalization。
旧 JSON 的 `mixed.alpha_mode: fixed` 应更新为 `bifocalrl`。

训练日志包含 `bifocal_mv_weight/snr/snr_threshold/rollout_keep/retained_fraction/query_weight`。
新增 `bifocal_sc_cluster_mean/sc_cluster_size/sc_within/sc_adjusted`，
分别记录保留簇均值、保留簇大小、簇内残差、调整后的 SC；过滤位置记为 0。
`bifocal_sc_between_gamma` 记录本次分解系数，与 PRISM 的 `bifocal_gamma` 区分。
`bifocal_answer_cluster` 是 query 内的答案簇 ID，随 rollout 传递并按原顺序恢复。
直接调用算法 API 时通过 `metadata={"answer_cluster_ids": ids}` 传入 `(P,G)` 的
`torch.long` 张量，各题 ID 范围为 `[0,G)`；只有 gamma 为 1 时可以省略。
可靠性计算仅保留在独立的 `reliability_sc_corr` baseline 中。

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
| `KL_COEF` | 默认 0；`circa` recipe 默认 0.005 并启用 reference KL loss |
| `PROMPT_SPLIT`, `APPLY_CHAT_TEMPLATE` | train、1 |
| `EVAL_DATA`, `EVAL_SPLIT`, `EVAL_STEPS` | 未设置、train、10 |
| `PRM_URL`, `PRM_MODEL` | 独立 GenPRM 服务及匹配的 tokenizer |
| `PYTHON` | python |

## CIRCA 与 KL-Cov

```bash
bash examples/bifocalrl/run.sh circa
```

按论文 Eq.2，答案按数学等价关系聚类，每条可解析回答得到本簇人数 / 可解析回答总数，
再加 0.1 的 boxed-format 奖励；无法抽取答案的回答为 0，全部失败时整组为 0。
这个 cluster/format 通道与 SCE 通道分别按问题做 sample-std z-score，再各乘 0.5 相加
（Eq.4–5），不额外执行 BifocalRL 的 batch 标准化或 query weighting。
Actor 缓存的 SCE 比 `KL(U||p)` 多一个 `log(V)` 常数，组内标准化后消去。

KL-Cov 按当前 token logprob 与优势的中心化乘积排序（Eq.6–8），选最高 2% 的有效
回答 token。选中位置用非裁剪的策略项加 `0.05 * abs(logp_current-logp_old)`，
其余位置使用普通 PPO clipping；reference KL 沿用 OpenRLHF 的独立 loss。
选择依据不参与求导，排除 prompt/padding；同分按 DP rank、token 顺序确定。
离散选择数为 `max(1, floor(k*N))`；`k=0` 时完全关闭 KL-Cov，恢复普通裁剪损失。

选择范围为同一次训练 forward 的全部 DP rank 的有效 token，支持各卡长度不同或某卡
没有有效 token。梯度累积沿用原生 loss 归一化，但不会合并不同 micro-batch 的协方差
候选池；这会影响与使用其他选择范围的实现之间的数值对齐。增大 micro batch 可扩大候选池。
仅支持 token-mean PPO，不叠加 dual clipping、GSPO 或额外 importance-sampling correction。

配置字段：`circa_cluster_weight=0.5`、`circa_certainty_weight=0.5`、
`circa_format_weight=0.1`、`kl_cov_fraction=0.02`、`kl_cov_coef=0.05`。
自定义 CLI / JSON 启动时，要匹配论文 reference KL，可显式传入
`--algo.kl.use_loss --algo.kl.init_coef 0.005`。启动脚本的这些默认值按 `circa`
recipe 名选择；设置 `BIFOCAL_CONFIG` 覆盖其他名字的配置时，应自行传入 reference KL 参数。
本实现依据论文公式独立编写；论文链接当前尚未提供官方代码。

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

CoVo 使用论文 Con、Vol、连续向量聚合和 Eq.10。
其中 token 概率分布解释为该步观测 token 的概率在位置维归一化。
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
BifocalRL 测试覆盖固定阈值、跨 query 过滤独立性、SNR/hard filtering、保留比例重加权、masked batch
normalization、全过滤/单保留/平坦奖励、被过滤回答的零策略梯度，以及重排后的统计量。
CIRCA 测试覆盖可解析回答分母、少数簇奖励、GDPO 优势、KL-Cov 损失和梯度，
以及两进程 Gloo 下的 DP 全局选择、空 rank 与实际 trainer training_step 的反向接线。
纯 CPU 测试使用轻量边界替身，不声称启动过实际 vLLM 或 DeepSpeed 模型。

完整上游测试命令已尝试；当前 CPU 环境缺少 DeepSpeed/datasets，并且 Transformers
早于此 fork 所需版本，部分上游模块无法加载。GPU 训练、真实 GenPRM 服务及论文性能
仍需在完整 GPU 环境中验收。
