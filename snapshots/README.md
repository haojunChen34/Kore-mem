# 所选回答轮次的冻结记忆

本目录保存 56 个场景在回答前的长期记忆逻辑快照，共 1,603 条记忆和 1,982 个版本记录。对应的 224 个回答位于 [per_question.json](../reports/memprobe/per_question.json)。

每个 `<场景 ID>.json` 包含原 SQLite 的 `memories` 和 `memory_versions` 两张表，按 `id` 排序导出。`columns` 是原列名，`rows` 是原列值；其中 `data_json` 保留数据库原始字符串，解析后可查看正文、摘要、结构化事实、来源证据、状态、裁决、关系和时间字段。没有根据答案修补记录。

[manifest.json](manifest.json) 逐场景记录原数据库哈希、原冻结时间、历史轮数、逻辑指纹、导出文件哈希和记录数。导出重新计算的逻辑指纹与原快照一致。文件中的原有时间原样保留；历史回放使用合成时钟，`original_frozen_at` 为实际冻结记录时间，不能把写入时间解释成事实发生时间。

这是两张记忆表的精确逻辑导出，不是原 SQLite 文件的完整复制。模型设置、密钥、本机路径、核心记忆、会话表、运行日志和答案表不包含在内，不能将此 JSON 直接替换为应用数据库。

[reanswer.py](../evals/memprobe/reanswer.py) 支持将这些 JSON 恢复到独立测试数据库后重新回答。它检查文件哈希、列结构及逻辑指纹，原样插入两张表，并调用产品代码重建派生搜索索引；不调用提取、裁决或维护模型，也不修改源快照。每题再复制一份数据库，读取强化只作用于该题副本。回答时钟按对应场景的 `history_user_turns` 恢复，保留记忆中的原有时间。

恢复后验证的是记忆逻辑指纹，生成的 SQLite 字节哈希不必等于 `original_db_sha256`。索引与回答行为取决于实际指定的产品版本，运行目录会固定源码、适配器、模型配置及快照哈希。操作命令见[使用已有记忆重新回答](../TEST_REPORT.md#snapshot-answers)。

快照由 MemProbe 数据派生，遵守上游 [CC BY-NC 4.0](../evals/memprobe/vendor/DATA_LICENSE)。
