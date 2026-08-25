# T8star-Aix IndexTTS 2.5 路线图

本路线图同时记录 ComfyUI 节点和桌面整合包的共同方向。项目只支持 **IndexTTS 2.5**。ComfyUI 基础依赖不强装 DeepSpeed、FlashAttention、Triton；桌面 v0.11.0 则内置与固定 Python/torch/CUDA ABI 匹配的可选轮子，仍不默认启用。

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
- [x] DeepSpeed 0.17.5 FP32/BF16 真实 WAV 回归，并修正 BF16 被误传成 FP16。
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

## v0.8.1（实现完成，待发布验收）

- [x] GPT 合成提示 KV Cache 根修复，长文本/多停顿块不再保护性关闭加速。
- [x] ASR 支持 OpenAI Whisper 与可选 faster-whisper 后端。
- [x] 简繁体、数字和标点归一化；中文/日文 CER、英文/西语/阿语 WER。
- [x] 差异明细、词级时间戳以及 ComfyUI 独立时间戳 JSON 输出。
- [x] `asr` / `asr-fast` 两组可选依赖，普通 TTS 安装仍不需要 ASR。
- [x] 当前环境 57 项通过、1 项按环境跳过；Transformers 4.57.6 结果相同。
- [x] V3 schema 12 项及 22 组 UI/API 工作流复核通过。
- [ ] Comfy Registry v0.8.1 发布复核（推送 main 后执行）。

## 依赖政策

- 普通生成路径不要求安装任何加速附加包。
- DeepSpeed 永不默认启用、永不自动安装，用户手动安装成功后才可选择。
- GPT 加速遇到不兼容采样参数时，单次任务自动走普通 GPT 路径。
- 当前官方 TensorRT 后端只支持 IndexTTS 2.0，因此本项目不宣称支持 2.5 TensorRT。
- vLLM-Omni 更适合作为后续 Linux/服务端 sidecar，不污染 Windows/ComfyUI 基础环境。

## 验收记录

- [x] 公共解析、时间轴、混音和加速探测测试通过。
- [x] V3 节点 schema 与 27 组 UI/API 工作流测试通过。
- [x] 当前基础环境：73 项测试通过。
- [x] Transformers 4.57.6：50 项测试通过，2 项按环境条件跳过。
- [x] IndexTTS 2.5 真实模型联合冒烟：两段生成、前导/段间停顿、目标时长和人声后处理通过。
- [x] 普通 BF16 与 BigVGAN CUDA kernel 完成真实 GPU 推理。
- [x] 缺依赖、初始化失败和运行失败回退测试通过。

## 后续候选

- vLLM-Omni Linux/服务端 sidecar，用于并发和吞吐优先场景。
- 波形时间轴拖拽吸附；v0.11.0 已提供波形、逐字标记和逐句毫秒级时间轴编辑。
