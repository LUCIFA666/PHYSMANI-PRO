# 小型证据索引

- `status_20261123.json`：整理时依据本地文件核对的当前状态，优先于旧报告中的运行状态。
- `original/`：原始小型报告、单元测试报告和部分进程退出回执，保留原字节。报告内的绝对路径指向原机器，不保证在 clone 中存在。
- `external_files_not_included.json`：未上传权重和关键数值依赖的身份。
- `source_provenance.json`：原样复制源文件和报告的来源及 SHA256。
- `export_validation.json`：在导出目录实际执行的核心单元测试结果。

方块模型完整成绩、TRAIN 专家目标下的执行器诊断、生命周期证据缺失的 worker 结果必须分开解读。保留一个 `worker/result.json` 不意味着补齐了对应 owner 的 wait／回收凭证。
