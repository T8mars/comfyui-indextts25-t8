# T8star-Aix IndexTTS 2.5 路线图

本路线图同时记录 ComfyUI 节点和桌面整合包的共同方向。项目只支持 **IndexTTS 2.5**；DeepSpeed、FlashAttention、Triton 等均为可选加速依赖，不进入基础安装。

## v0.5.0（已实现）

- [x] 官方模型清单同步到固定 revision，校验配置文件大小与 SHA256。
- [x] 中文多音字、英文 CMU 音素、日文假名发音控制。
- [x] 角色音色、可连接的角色库与缺失角色预检查。
- [x] `[角色] 台词`、`角色: 台词`、`角色|台词|语言|时长系数` 和 JSON 批量格式。
- [x] SRT BOM/CRLF、多行字幕、逗号或点毫秒格式解析。
- [x] SRT 顺延/时间轴混音，以及限制在 0.5–2.0 的二次时长适配。
- [x] 合并音频、逐句 `AUDIO` 列表和机器可读 JSON 时间轴报告。
- [x] 关闭、自动安全、BigVGAN CUDA、Torch Compile、GPT 加速、DeepSpeed 模式。
- [x] 可选加速缺失、初始化失败或运行失败时回退普通模式。
- [x] 14 组 UI/API 示例工作流。
- [x] Transformers 4.52.1 与 4.57.6 兼容测试。

## 依赖政策

- 普通生成路径不要求安装任何加速附加包。
- DeepSpeed 永不默认启用、永不自动安装，用户手动安装成功后才可选择。
- GPT 加速遇到不兼容采样参数时，单次任务自动走普通 GPT 路径。
- 当前官方 TensorRT 后端只支持 IndexTTS 2.0，因此本项目不宣称支持 2.5 TensorRT。
- vLLM-Omni 更适合作为后续 Linux/服务端 sidecar，不污染 Windows/ComfyUI 基础环境。

## 验收记录

- [x] 公共解析、时间轴、混音和加速探测测试通过。
- [x] V3 节点 schema 与 14 组 UI/API 工作流测试通过。
- [x] Transformers 4.52.1：35 项测试通过。
- [x] Transformers 4.57.6：35 项测试通过。
- [x] 普通 BF16 与 BigVGAN CUDA kernel 完成真实 GPU 推理。
- [x] 缺依赖、初始化失败和运行失败回退测试通过。

## 后续候选

- vLLM-Omni Linux/服务端 sidecar，用于并发和吞吐优先场景。
- 官方流式接口的边生成边播放适配。
- ASR 自动校对、字幕回写和可视化时间轴编辑。
