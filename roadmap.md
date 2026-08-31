# T8star-Aix IndexTTS 2.5 路线图

本路线图同时记录 ComfyUI 节点和桌面整合包的共同方向。项目只支持 **IndexTTS 2.5**。当前公开基线为
**ComfyUI Node 0.22.0 / Desktop 0.23.0**。
ComfyUI 基础依赖不强装 DeepSpeed、FlashAttention、Triton；桌面整合包内置与固定 Python/torch/CUDA
ABI 匹配的可选轮子，仍不默认启用。

## v0.21.4

- [x] 将 NVIDIA `waves_per_eu` 报错识别为 PyTorch/Triton 加速兼容问题，并保留原始错误详情。
- [x] 加速运行失败后释放缓存模型，以普通模式自动重试；精确错误回归测试覆盖该路径。
- [x] Desktop 流式试听同步接入回退，高显存显卡不再默认选择实验性 GPT 加速。
- [x] 刷新全部 33 组 UI/API 工作流并完成完整发布门禁。

## v0.22.0 / Desktop v0.23.0

- [x] 为中文数字、日期和年份归一化加入独立可选依赖与真实 `1939年 → 一九三九年` 环境自检。
- [x] 后端缺失或自检失败时安全保留原文；节点工具提示和中英文安装说明同步更新。
- [x] 版本提升后重建并校验全部 33 组 UI/API 工作流。

- [x] 新增中、英、日、西、阿五语言真实模型回归，报告 CER/WER、分段语速、削波、静音、时长、RTF 与峰值显存。
- [x] 支持与旧 `quality-report.json` 对比，严格模式在明确音质或性能回退时返回失败状态。
- [x] Desktop 把 `segment_rate_guard` 显示为语速柱状图和可读表格，并分别保存原始段、自动重试候选与当前采用段。
- [x] 用户可只重做选中内部段并重新合并最终 WAV；首次完整结果保留在受路径约束的工作区中。
- [x] `indextts.cli` 使用正式 IndexTTS 2.5 五语言推理，暴露情感、时长、采样、CFM、精度和可选加速参数，并通过真实 BF16 WAV 冒烟。
- [x] 将旧“尚未开发”规划改为历史架构文档，并同步当前 Node/Desktop 状态与使用示例。
- [x] RTX 真实权重生成 5 组固定 WAV：严格门禁通过、全部无削波、RTF 0.552–0.864、峰值显存不超过 5.77 GB。
- [x] 根仓库 155 项、节点 172 项测试通过（另 1 项按环境跳过）；compileall、Pylint Fatal、Comfy 节点校验、Electron 打包与包内运行时自检通过。

## v0.21.2

- [x] Arabic ASR 固定为 Whisper `small`，同一 WAV 的 WER 从 0.6154 降至 0.1923；其余语言继续使用 `base`。
- [x] Arabic CER/WER 比较加入保守字形规范化；GPT 推理迁移到 Transformers `DynamicCache` 并保留旧 tuple 兼容。
- [x] Torchaudio 2.9+ 增加 TorchCodec/FFmpeg 共享 DLL 启动预检，便携 Torchaudio 2.8 路径保持不变。
- [x] RTX 真机完成 8GB/24GB 五语言双档回归，峰值约 3.34/5.52 GiB；新增两份脱敏基线。
- [x] 定时任务新增 JSON/Markdown/SVG 历史趋势产物，跟踪 CER/WER、RTF 与峰值显存。

## v0.21.1

- [x] 固定 OpenAI Whisper `20250625` 与 `base` CUDA 质量配置，生成中英日西阿真实 CER/WER 脱敏 GPU 基线。
- [x] 新增独立的每周/手动自托管 GPU 工作流，普通 CI 不下载或加载大模型。
- [x] GPT 推理类显式继承 `GenerationMixin`，消除 Transformers 4.50+ 的继承风险。
- [x] Torchaudio 2.9+ 使用原生 TorchCodec I/O，保留 2.8 便携运行时兼容和清晰 ABI 错误提示。
- [x] 普通 CI 增加 torch/torchaudio 2.9 + TorchCodec 0.9 矩阵，版本提升后刷新全部 33 组 UI/API 示例。

## v0.21.0

- [x] 对每个内部长文本分段采集真实时长和语言感知文字单位，形成稳健的前序中位语速基线。
- [x] 仅在后续分段语速低于基线 45% 时单独重做该段；候选没有明显改善则保留原音频。
- [x] 短句、一般情绪放慢、确定性采样和原生目标时长模式不自动重做。
- [x] 生成状态写入可审计的 `segment_rate_guard` 报告，并覆盖选择性重做的单元测试。
- [x] 0.20.9 已在 Comfy Registry 正式激活，原 Publish 任务幂等重跑后全部成功。

## v0.20.9

- [x] 模型下载改为当前进程内 Hugging Face 分块传输，保留进度、中止、续传和完整性校验。
- [x] 新增 Pylint 4 Fatal/Astroid 双平台 CI 门禁，Registry 扫描器兼容问题可在发布前复现时被拦截。
- [x] Linux Python 3.11、Windows Portable Python 3.10 / torch 2.8、打包、上传与 Registry 激活全部完成。

## v0.20.5

- [x] 删除 audio.cpp 一键联网安装节点及其运行时/GGUF 下载代码，解决 Registry 联网请求标记。
- [x] 保留只接受本地绝对路径的 audio.cpp 实验生成节点，不改变默认 Python 推理路径。
- [x] 删除第 34 组一键安装示例，刷新其余 33 组 UI/API 工作流。

## v0.20.2

- [x] 将 ComfyUI V3 节点架构测试从脆弱的列表下标改为稳定节点 ID，覆盖新增节点后的真实顺序。

## v0.20.1

- [x] 修复 Linux CI 中平台模拟污染 Python 全局 `os.name`、导致 pytest 无法构造路径的问题。

## v0.20.0

- [x] 新增读取 Desktop `.t8voice.zip` 的已保存音色节点，保留角色默认语言、情感与元数据。
- [x] 新增主动确认、断点续传、磁盘预检和 SHA-256 校验的 audio.cpp Windows 运行时/GGUF 一键组件。
- [x] audio.cpp 生成路径留空时自动读取已安装组件，手动绝对路径继续优先。
- [x] 新增第 33、34 组 UI/API 示例并将全部工作流版本刷新到 0.20.0。

## v0.19.0

- [x] 缺失模型自动下载把现有文件扫描、逐文件传输和 SHA-256 校验映射到 ComfyUI 任务进度。
- [x] Hugging Face 下载保留断点续传，哈希损坏文件强制刷新；ComfyUI 显示总体任务进度，终端显示当前文件、阶段、速度和剩余时间。
- [x] 下载前执行目标磁盘空间预检，安全下限不足时提前报出可操作错误。
- [x] Desktop 0.20.0 同步提供可视化下载卡片、当前文件、总进度、速度、ETA、磁盘状态和精确失败原因。
- [x] 推送前刷新全部 32 组 UI/API 示例并执行完整测试、compileall、V3 schema 和 Registry 包校验。

## v0.18.0

- [x] 长英文、西语和长 SRT 逐句检测 `max_mel_tokens` 与异常音频时长，必要时缩短分段并重试一次。
- [x] 真实 IndexTTS 2.5 长英文/西语 WAV 回归，并修复 Windows GBK 西语日志编码中断。
- [x] 参考条件缓存状态包含条目、容量、命中、未命中和命中率；安全清理只删除本节点缓存。
- [x] 示例 06、13、26 覆盖长拉丁语系、长 SRT 与参考缓存管理，推送前刷新全部 32 组 UI/API 工作流。
- [x] 本版不扩展 GPU 压测矩阵，也不增加 Linux sidecar 服务。

## v0.17.0

- [x] 上下文逐句情感建议：区分前文、目标句、后文与不同角色。
- [x] 建议八维情感向量和强度，默认保留已有人工逐句情感。
- [x] 节点只输出可编辑台词脚本/JSON，不生成音频。
- [x] 第 32 组 UI/API 工作流先分析与预览，确认前不连接生成节点。
- [x] 完整测试、Comfy V3 schema 与 Registry 包校验。
- [x] 版本提升到 0.17.0，并在推送前刷新全部 32 组 UI/API 示例。

## v0.11.0（本地实现）

- [x] 修复 ComfyUI 2026.08 不再自动把自定义节点目录加入 `sys.path` 时的内置 `indextts` 导入。
- [x] 单句与多角色支持 ASR 低分自动换 seed 重试并保留最佳结果。
- [x] 参考音频质量评分、静音裁剪、高信息片段选择和波形预览。
- [x] ASR 波形/逐字时间标记，多角色时间轴显示逐字标记。
- [x] 生成后、空闲、连续 N 次生成和手动模型释放，且不清理其他 ComfyUI 模型。
- [x] 隔离的 audio.cpp IndexTTS2.5 GGUF 实验节点；不捆绑 CLI 与 GGUF，不改变默认推理路径。
- [x] 扩展为 27 组 UI/API 示例工作流并按 ComfyUI V3 实时 schema 校验。

## v0.10.0（本地实现）

- [x] 角色音色节点明确提供“该角色默认情感”连接点。
- [x] 角色音色/情感合并与 `Merge Voice Emotions` 兼容节点支持 1–16 个独立角色配置。
- [x] 新增多角色独立情感 UI/API 示例，并验证情感不会跨角色串用。
- [x] 桌面角色音色库支持跟随音色、情感参考音频、八维向量和文本描述四种模式。

## v0.9.0（本地实现与验证完成）

- [x] Windows Python 3.10 / torch 2.8.0+cu128 精确轮子清单与 SHA-256。
- [x] FlashAttention 2.8.3 + Triton 3.4.0.post21 的真实 GPT 加速 WAV 回归。
- [x] DeepSpeed 0.17.5 FP32/BF16 真实 WAV 回归；Linux 保持 BF16，Windows BF16 请求使用兼容性更好的 FP16 workspace 并保留失败回退。
- [x] synthetic-prompt GPT KV Cache 防误命中，避免重复长提示跨缓存块时断言失败。
- [x] 去除音色缓存未命中路径的重复 Wav2Vec 前向、文件读取与 16 kHz 重采样。
- [x] CI 覆盖 Linux 当前环境与 Windows Portable Python 3.10 / torch 2.8 基线。
- [x] Registry 发布幂等保护：任务串行化，精确版本预检查，已存在版本绿色跳过。
- [x] 普通、低显存、流式和后处理输出统一做 20ms 尾部淡出并归零。
- [x] 官方问题 #792 的中文整词标注提示、音节数校验和固定 seed 音频回归。

## v0.6.0（已实现）

- [x] 官方模型清单同步到固定 revision，校验配置文件大小与 SHA256。
- [x] 中文多音字、英文 CMU 音素、日文假名发音控制。
- [x] 角色音色、可连接的角色库与缺失角色预检查。
- [x] `[角色] 台词`、`角色: 台词`、`角色|台词|语言|时长系数` 和 JSON 批量格式。
- [x] SRT BOM/CRLF、多行字幕、逗号或点毫秒格式解析。
- [x] SRT 顺延/时间轴混音，以及限制在 0.5–2.0 的二次时长适配。
- [x] 合并音频、逐句 `AUDIO` 列表和机器可读 JSON 时间轴报告。
- [x] 关闭、自动安全、BigVGAN CUDA、Torch Compile、GPT 加速、DeepSpeed 模式。
- [x] 可选加速缺失、初始化失败或运行失败时回退普通模式。
- [x] 语言感知自动分段与不加载权重的 Token/停顿预览。
- [x] 标点停顿预设、自定义毫秒停顿和 `<pause=0.5>` 显式标记。
- [x] GPT 加速长文本/多语音块 KV Cache 风险自动保护。
- [x] 目标秒数自然适配、补静音和强制精确裁剪模式。
- [x] 五种可选人声后处理及独立 AUDIO 节点。
- [x] 18 组 UI/API 示例工作流。
- [x] Transformers 4.52.1 与 4.57.6 兼容测试。

## v0.7.0（已发布）

- [x] CFM 扩散步数、CFG 引导强度、温度参数贯通节点与正式推理核心。
- [x] `target_duration` 原生一次推理控制，外加停顿按总时长预算计算。
- [x] 单句生成与 SRT 多角色生成支持 `native`，旧核心自动回退二次推理。
- [x] CFM 高级参数示例及 19 组 UI/API 工作流。
- [x] 当前环境 50 项测试通过、2 项按环境跳过；Transformers 4.57.6 隔离环境结果相同。
- [x] 真实 IndexTTS 2.5 权重完成固定 seed、自定义 CFM、流式返回及原生目标时长联合冒烟。
- [x] README、19 组 UI/API 示例工作流、版本号和发布清单复核完成。

## v0.8.0（实现完成）

- [x] 本地 Whisper ASR 自动校对，输出识别文本、编辑距离、CER、相似度与阈值判定。
- [x] 多角色逐句 ASR 报告及按“全部/仅通过”策略回写字幕文本。
- [x] 原始 SRT 时间和生成音频真实时间轴两种字幕时间来源。
- [x] 毫秒级时间轴 JSON 编辑节点，并输出标准 IMAGE 彩色轨道预览。
- [x] 新增 ASR 校对、时间轴编辑、字幕回写示例，扩展为 22 组 UI/API 工作流。
- [x] ASR 作为 `asr` 可选依赖；普通 IndexTTS 推理不要求安装 Whisper。
- [x] 当前环境与 Transformers 4.57.6 均为 53 项通过、2 项按环境跳过；V3 节点和 22 组工作流审计通过。
- [x] 真实 IndexTTS 音频完成 Whisper tiny/base CUDA 校对，失败阈值与字幕保留原文策略符合预期。

## v0.8.1（历史版本，已发布验收）

- [x] GPT 合成提示 KV Cache 根修复，长文本/多停顿块不再保护性关闭加速。
- [x] ASR 支持 OpenAI Whisper 与可选 faster-whisper 后端。
- [x] 简繁体、数字和标点归一化；中文/日文 CER、英文/西语/阿语 WER。
- [x] 差异明细、词级时间戳以及 ComfyUI 独立时间戳 JSON 输出。
- [x] `asr` / `asr-fast` 两组可选依赖，普通 TTS 安装仍不需要 ASR。
- [x] 当前环境 57 项通过、1 项按环境跳过；Transformers 4.57.6 结果相同。
- [x] V3 schema 12 项及 22 组 UI/API 工作流复核通过。
- [x] Comfy Registry v0.8.1 发布复核。（历史记录）

## 依赖政策

- 普通生成路径不要求安装任何加速附加包。
- DeepSpeed 永不默认启用、永不自动安装，用户手动安装成功后才可选择。
- GPT 加速遇到不兼容采样参数时，单次任务自动走普通 GPT 路径。
- 当前官方 TensorRT 后端只支持 IndexTTS 2.0，因此本项目不宣称支持 2.5 TensorRT。
- vLLM-Omni 更适合作为后续 Linux/服务端 sidecar，不污染 Windows/ComfyUI 基础环境。

## 验收记录

- [x] 公共解析、时间轴、混音和加速探测测试通过。
- [x] V3 节点 schema 与全部 32 组 UI/API 工作流测试通过。
- [x] 当前基础环境：73 项测试通过。
- [x] Transformers 4.57.6：50 项测试通过，2 项按环境条件跳过。
- [x] IndexTTS 2.5 真实模型联合冒烟：两段生成、前导/段间停顿、目标时长和人声后处理通过。
- [x] 普通 BF16 与 BigVGAN CUDA kernel 完成真实 GPU 推理。
- [x] 缺依赖、初始化失败和运行失败回退测试通过。

## 后续候选

- vLLM-Omni Linux/服务端 sidecar，用于并发和吞吐优先场景。
- 波形时间轴拖拽吸附已由 Desktop 提供；下一阶段只考虑更细的 ASR 音素级吸附，不重复建设现有功能。
