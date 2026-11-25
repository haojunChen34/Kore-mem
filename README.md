# Kore-mem

**KoreAgent 长期记忆的 MemProbe 测评与可核验结果。** 本仓库包含固定数据、真实产品适配器、所选完整轮次的 224 个回答，以及对应的 56 份冻结记忆逻辑快照。

产品实现位于 [Kore-Agent](https://github.com/haojunChen34/Kore-Agent)。测试直接调用真实产品的提取、保存、冲突裁决、检索和读取服务，不另建记忆实现。适配器使用说明见 [详细测试说明](TEST_REPORT.md#running)。

## 测试结果

**测试结果：85.27%（191/224）**，按 MemProbe 原生规则计分。suite56 的 56 个场景、224 个问题均完成；15 题部分正确、18 题错误，无缺失回答。

| 问题类型 | 题数 | 正确 | 部分正确 | 错误 | 正确率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 当前事实 | 56 | 46 | 0 | 10 | 82.14% |
| 历史事实 | 28 | 28 | 0 | 0 | 100.00% |
| 变化检测 | 56 | 52 | 0 | 4 | 92.86% |
| 冲突值 | 28 | 24 | 4 | 0 | 85.71% |
| 来源识别 | 34 | 24 | 7 | 3 | 70.59% |
| 时间关系 | 22 | 17 | 4 | 1 | 77.27% |

回答阶段实际累计输入 **4,792,853 Token**、输出 **138,369 Token**，共 **860 次模型请求、1,042 次记忆相关工具调用**；平均每题耗时 25.33 秒。规则评分无模型调用。模型为 `gpt-5.6-sol`，来自 OpenAI 兼容服务。

查看 [详细测试说明与结果](TEST_REPORT.md)、[全部指标](reports/memprobe/metrics.json)、[224 题回答](reports/memprobe/per_question.json)、[冻结记忆](snapshots/README.md)和[实际配置](reports/memprobe/configuration.json)。完整报告另含四类范式、更新与保持能力、分散信息、结构化子项、P50/P95 耗时、工具行为、逐场景成绩及记忆构建成本。

## 测试流程与范围

```text
逐场景创建独立环境
→ 按原角色和顺序回放历史
→ 真实产品提取、自动保存、关系裁决和维护
→ 冻结长期记忆
→ 每题从独立快照副本开始
→ BM25 检索最多 5 条摘要，按需搜索及读取全文
→ 保存回答、API 用量、工具和耗时
→ 固定上游规则评分
```

- 历史助手回复直接回放；提取阶段不可见问题、标准答案及评分标签，不按答案补写记忆。
- 开启自动记忆、运行后反思和自动保存；冲突处理、提取复核、权限过滤使用真实产品逻辑。
- 回答为单组原生策略，允许 `tool_catalog`、`memory_search`、`memory_list`、`memory_read`；每题结束后不回写学习。
- 原始会话全文、核心记忆、个人档案、Skill、工作区文件和外部服务不作为额外答案来源。这是长期记忆专用配置。
- 回放时钟从 2026-01-01 起，每用户轮递增一分钟；周期老化间隔 360 轮。现有场景未达到该阈值，不能证明长时老化效果。

上游数据与评分固定到 [`9068d8d`](https://github.com/jq-ding/MemProbe/tree/9068d8d9a2a47268fe6f7601b99a56a283c37be5)，[文件哈希](evals/memprobe/vendor/manifest.json)可核验。`partial` 计入总题数但不计入正确数；原生整题 `correct` 与全部子字段正确并非同义，报告保留子项。

结果适用于这批已知基准题。未独立标注检索相关性，未设置成本对照，因此不报告 Recall@5、nDCG@5、事实保存覆盖率或 Token 降幅，也不据此推断未见数据表现。

## 复现与核验

环境配置、运行命令与离线核验见 [详细测试说明](TEST_REPORT.md#running)。

| 方式 | 做什么 | 是否调用模型 |
| --- | --- | --- |
| 离线核验 | 对公开的 224 个回答重新判分、核对指标与快照 | 否 |
| [使用已有记忆重新回答](TEST_REPORT.md#snapshot-answers) | 恢复公开 JSON 快照，真实检索与读取，生成新回答并评分；支持单题、单场景及全量 | 是，仅回答阶段 |
| 从历史重建 | 重新提取和保存记忆，然后回答并评分 | 是，包含记忆构建 |

第二种方式的入口是 `evals/memprobe/reanswer.py`。每题从独立记忆副本开始，不重新提取、不回写学习；新回答和 JSON 指标保存在新的运行目录。它验证已有记忆上的回答表现，不评价新的提取效果，也不保证新成绩与本文相同。

## 文件与许可

仓库共 96 个文件，按以下结构归档：

```text
README.md                 项目简介和结果摘要
TEST_REPORT.md            详细测试说明与完整结果
configs/                  运行配置模板
evals/memprobe/           测试代码及固定上游依赖
reports/memprobe/         逐题回答、指标、实际配置
snapshots/                56 份记忆快照及其说明和哈希
scripts/                  离线核验入口
```

在仓库根目录运行 `python -X utf8 -B scripts/validate_package.py`，可离线复算已公开答案的评分，并检查记忆指纹、上游哈希和文件范围。

从历史重建的运行用 `evals/memprobe/aggregate_results.py` 汇总 JSON；快照重新回答的运行由 `reanswer.py` 自动汇总，并通过同一入口的 `--verify` 离线复核。测试代码不自动生成 Markdown 报告或简历说明；本仓库的说明文档单独维护。

快照为原 `memories` 与 `memory_versions` 两表的精确逻辑导出，保留正文、证据、状态、时间和版本，未附带日常数据库或模型配置。导出范围与来源见 [归档说明](TEST_REPORT.md#archive)及 [configuration.json](reports/memprobe/configuration.json) 的 `publication_provenance` 字段。

本项目适配器采用 [MIT License](LICENSE)。MemProbe 来源为 [jq-ding/MemProbe](https://github.com/jq-ding/MemProbe)，上游代码采用 [MIT](evals/memprobe/vendor/LICENSE)，数据采用 [CC BY-NC 4.0](evals/memprobe/vendor/DATA_LICENSE)。复制数据及其派生的问题、回答和记忆快照继续遵守数据许可，不因仓库 MIT 许可而获得商业使用授权。
