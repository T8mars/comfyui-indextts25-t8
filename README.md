# comfyui-indextts25-T8

**简体中文** | [English](README_EN.md)

IndexTTS 2.5 的 ComfyUI V3 原生节点集成。节点菜单位于：

`T8star-Aix / Audio / IndexTTS 2.5`

作者标识：**B 站：T8star-Aix**。

## 作者与资源链接

- B 站：[T8star-Aix](https://space.bilibili.com/385085361)
- YouTube：[T8star-Aix](https://www.youtube.com/@T8star-Aix/)
- Hugging Face：[t8star](https://huggingface.co/t8star)
- IndexTTS 2.5 完整模型：[t8star/IndexTTS-2.5-Comfy](https://huggingface.co/t8star/IndexTTS-2.5-Comfy)
- API 注册（推广链接）：[api.seedance.nz](https://api.seedance.nz/sign-up?aff=5f4w)
- 在线 AI 应用（RunningHub）：[T8star-Aix 用户主页](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- ComfyUI 整合包：[夸克网盘下载](https://pan.quark.cn/s/264edb7e36bd)
- IndexTTS 2.5 模型网盘：[夸克网盘下载](https://pan.quark.cn/s/c9c267081fbf)

本目录固定使用 IndexTTS 2.5 推理核心和正式 2.5 模型清单，不会回退或误载 IndexTTS 2.0。

当前版本基线：**ComfyUI Node 0.16.3 · Desktop 0.16.0 · Core `ee40fa7d` · Upstream Model `c39ce5ba`**。

### v0.16.3 Registry 安全扫描兼容

- 保留 audio.cpp 实验节点，改用 ComfyUI V3 可等待的异步、无 Shell 固定参数进程接口，并继续执行绝对路径、后端和参数白名单校验。
- 手动更新检查只允许访问代码内置的三个官方固定端点，禁用跨站重定向并限制响应不超过 1 MiB；节点版本改由 GitHub Contents API 读取。
- 保留捆绑上游语言表的完整映射，同时改写一个会被扫描器误判为系统提权命令的语言代码字面量；所有语言索引和行为不变。
- Registry 发布包通过 `.comfyignore` 排除测试和 CI 文件；GitHub 源码、完整测试与示例仍全部保留。

### v0.16.2 模型仓库新名称

- Hugging Face 完整模型仓库已更名为 `t8star/IndexTTS-2.5-Comfy`，下载器、模型清单和文档同步使用新地址。
- 仓库名称改变不影响 ComfyUI 标准目录；模型仍应直接放在 `ComfyUI/models/TTS/IndexTTS-2.5/`。

### v0.16.1 模型目录说明

- GitHub 中英文 README 与 Hugging Face 模型卡同步增加节点/模型分离目录树、Windows 完整路径和“多套一层目录”的错误示例。
- 完整模型仍放在 `ComfyUI/models/TTS/IndexTTS-2.5/`，节点代码放在 `ComfyUI/custom_nodes/comfyui-indextts25-T8/`。

### v0.16.0 完整模型单仓库与按需修复

- 新增完整模型仓库 `t8star/IndexTTS-2.5-Comfy`：在未修改原始权重的前提下，把官方 2.5 主模型、来自官方 2.0 仓库的必需 `bpe.model`，以及推理必需的 Wav2Vec2-BERT、CAMPPlus、BigVGAN 放在同一目录结构中。
- 模型加载器新增“缺失时自动下载/修复完整模型”；默认关闭，只有用户同时勾选许可证确认后才联网下载，约需 7.7 GiB。
- 下载器按固定 revision 与 SHA-256 清单检查并只修复缺失或损坏的文件，手动下载完整仓库也可直接使用。
- 官方 `IndexTeam/IndexTTS-2.5` 仓库目前没有 `bpe.model`，所以只下载该仓库会被本节点正确判定为目录不完整；这不是用户目录放错。

### v0.15.0 逐句情感与安全字幕槽位

- 同一角色可在每条批量台词、JSON、SRT 或时间轴记录中单独覆盖文本情感或八维情感向量；留空继承角色默认情感。
- 新增 `31_per_line_emotion.json`，完整演示同一个音色在平静、生气和角色默认情感间逐句切换。
- 字幕槽位安全默认值改为 `pad`，超长语音不再默认裁掉句尾；`native/exact` 明确标注裁剪风险。
- Desktop 0.16.0 的流式试听由内置 PyAV 编码，不再要求用户系统安装 FFmpeg/FFprobe。
- `duration_factor` 统一说明为官方声学时长适配，不再误导为自然语气语速。

0.14.0 新增当前模式真实性能基准、手动上游更新检查、跨模型重载的安全参考条件缓存，以及保留全部音频的多候选质量筛选；
Desktop 0.16.0 可依次实测加速模式并给出推荐，所有耗时或联网操作仍必须由用户主动触发。
完整模型仓库保留每个文件的上游来源与固定 revision；主权重仍是官方 2.5 文件，`bpe.model` 来自官方 2.0 模型仓库，未对权重内容做修改。
Desktop 与 Node 是两个独立发行物，因此各自使用独立版本号；Core/Model 是固定的官方代码和权重 revision。

### v0.14.0 真实性能、多候选与缓存更新

- “真实性能基准”使用相同参考音频、文本与 seed，预热后报告当前模型加载器实际生效模式的中位 RTF、最佳 RTF 和峰值显存；切换加载器模式后重跑即可公平比较。
- 语音生成的“追加候选数量”会保留全部候选，并输出 AUDIO 列表；本地 ASR 可用时结合台词相似度和音频质量选优，不可用时按削波、静音、直流偏移等技术指标选优，不会让已有音频失败。
- 模型加载器默认启用持久参考条件缓存。缓存按参考音频内容、模型 revision/指纹、精度和参考设备隔离，使用 `safetensors`，最多保留 128 项；高级选项可关闭。
- “检查更新”手动比较官方代码、Hugging Face 模型与本节点版本，只输出报告，不下载、不覆盖文件。
- Desktop 0.15.0 在启动前可顺序实测各加速模式并应用推荐；基准不会与正在运行的服务并发，也不会自动启动。

### v0.13.0 加速诊断更新

- “环境与可选加速”在不加载模型的情况下报告准确的已安装依赖版本，便于区分“没有安装”和“运行时初始化失败”。
- Desktop 启动器可重新检测 BigVGAN CUDA、Torch Compile、GPT 加速和 DeepSpeed，并逐项解释预计启用或回退原因。
- JSON 诊断报告包含系统、硬件、模型校验、所选运行参数、固定代码/模型 revision 与排错说明，方便用户直接随 Issue 提交。
- 预检只代表依赖具备；真正生效模式仍以启动日志和 WebUI 环境诊断为准。DeepSpeed 的 AIO/cuFile 可选扩展警告不影响语音推理判定。

### v0.12.0 低显存与精度更新

- `auto` 精度只在显卡原生支持时选择 BF16；旧显卡自动回退 FP16，也可手工选择 `float16`。
- 参考编码器新增 `auto / same / cpu`：低于 10GB 显存时，`auto` 会把 Wav2Vec/CAMPPlus 放到 CPU。
- 可选“快速默认情感”在没有独立情感输入时复用音色条件，减少一次参考编码；默认关闭以保持原行为。
- 环境节点新增显卡名称、显存、推荐精度、参考编码器位置和安全加速模式，不加载模型即可先检查。
- 新增第 28 组低显存工作流；Registry 发布后继续等待安全扫描，只有 Active 才算正式发布成功。

### v0.11.5 发布保护

- Registry 发布任务按仓库串行排队，避免手动触发与延迟 push 事件并发上传同一版本。
- 上传前通过官方只读版本接口检查精确版本；Active/Pending 等已存在状态会绿色跳过，只有 404 才允许上传。
- Registry 查询发生超时、5xx 或返回异常内容时重试并安全停止，不会在状态不明时盲目发布。

### v0.11.4 稳定性更新

- 首次使用或更换音色时只执行一次 Wav2Vec 编码和一次音频读取/重采样，减少重复的参考音频预处理。
- 模型加载信息明确显示 Node/Core/Model 版本，运行时节点版本直接读取 `pyproject.toml`，避免版本号漂移。
- CI 同时验证 Linux 当前环境与 Windows Portable 对应的 Python 3.10 / torch 2.8 环境。
- GPT 加速的 synthetic-prompt KV Cache 防误命中修复已核验并保留专门回归测试。

- 普通批量台词可直接包含官方 `<文字|读音>` 标注，不再被字段分隔符误切开。
- JSON/SRT 时间轴严格校验成对起止时间、先后顺序和安全范围；合成前会阻止超过 1 GiB 的异常密集分配。
- 单人和多角色 ASR 校对在后端缺失、下载失败或运行异常时都会保留已经生成的音频。
- 自动兼容 v0.10 模型加载器工作流的自定义路径错位；覆盖更新同目录权重后会使用新的文件指纹重载。
- “全部释放”同时清理本扩展的 IndexTTS 与 Whisper 缓存；audio.cpp 增加 Apple Metal 选项。
- Windows DeepSpeed 的 BF16 请求自动使用兼容性更好的 FP16 推理 workspace；失败仍安全回退普通模式。
- 27 组 UI 工作流增加控件顺序/类型校验，并在 Registry 发布前使用当前 ComfyUI V3 API 自动测试。
- 参考音频缓存改用标准 PCM16 WAV 写入，兼容最新版 `torchaudio` 未安装 TorchCodec 的环境。

## 已实现节点

1. `IndexTTS 2.5 模型加载器 · T8star-Aix`
   - 扫描标准模型目录和 `extra_model_paths.yaml` 中的 `TTS` 路径
   - 正式模型文件大小校验；可选完整 SHA-256 校验
   - `auto / CUDA / CPU` 设备和 `auto / bfloat16 / float16 / float32` 精度
   - `auto / same / cpu` 参考编码器位置，以及可选默认情感条件复用
   - 默认开启基于 `safetensors` 的持久参考条件缓存；高级选项可关闭
   - 全局惰性缓存、同模型线程锁、可选生成后释放或连续 N 次生成后安全重载
2. `IndexTTS 2.5 情感控制 · T8star-Aix`
   - 跟随音色参考
   - 独立情感参考音频
   - 八维情感向量（高兴、愤怒、悲伤、恐惧、厌恶、低落、惊讶、自然）
   - 文本情感描述；Qwen 情感模型按需加载到同一推理设备
3. `IndexTTS 2.5 采样设置 · T8star-Aix`
   - 稳定默认值、随机采样、beam、temperature、top-p/top-k
   - CFM 扩散步数、引导强度和噪声温度；默认保持官方 `25 / 0.7 / 1.0`
   - 按语言自动分段（EN/ES 60、AR 80、JA 100、ZH 120 Token）或手动上限
   - 标点停顿预设、自定义毫秒数、显式 `<pause=0.5>`、段间停顿与文本归一化
4. `IndexTTS 2.5 分段与停顿预览 · T8star-Aix`
   - 不加载神经网络权重，只读取正式模型 Token 词表
   - 输出每段 Token 数、语音块、段后停顿和 GPT 加速风险；文本可直接透传给生成节点
5. `IndexTTS 2.5 发音控制 · T8star-Aix`
   - 官方 `<文字|读音>` 标注格式
   - 中文数字声调拼音、英文 CMU 音素、日语假名校验
   - 工作流内嵌发音词典、长词优先、手工标注优先
   - 输出处理后文本和完整替换/校验报告，不修改模型全局状态
6. `IndexTTS 2.5 语音生成 · T8star-Aix`
   - 标准 ComfyUI `AUDIO` 输入/输出
   - 中、英、日、西、阿五种语言入口
   - 音色克隆、seed、官方 `duration_factor=0.5~2.0` 声学时长适配；它不是自然语气语速，极端值可能拉长或失真
   - 目标秒数支持原生长度调节器单次适配，以及自然二次适配、补静音、强制精确兼容模式
   - 可选人声清晰、清晰旁白、去刺耳、温暖、峰值归一化后处理
   - 可生成 1–4 个候选并从独立 AUDIO 列表输出；ASR 可用时结合文本相似度与波形质量选优，否则使用波形质量选优
7. `IndexTTS 2.5 角色音色 · T8star-Aix`
   - 将角色名、标准 AUDIO、默认语言和可选情感封装成工作流内音色
8. `IndexTTS 2.5 角色音色 / 情感合并 · T8star-Aix`
   - 自动增长输入，可汇总 1–16 个角色各自的音色与独立情感；重复角色名会在排队前报错
9. `IndexTTS 2.5 Merge Voice Emotions · T8star-Aix`
   - 对应社区常用的 `Merge Voice Emotions` 搜索名称，输出与“角色音色 / 情感合并”完全一致
   - 合并的是角色配置列表，不会把多个角色的八维情感数值混成一个新情绪
10. `IndexTTS 2.5 批量台词 / SRT · T8star-Aix`
   - 解析 `角色|台词|语言|时长系数|逐句情感`、JSON 数组和标准 SRT
   - 同一角色每句可用文本描述或八维向量覆盖情感；留空继续继承角色默认情感
   - 普通批量台词中的 `<文字|读音>` 会作为正文保留；复杂正文也可使用 JSON 格式
   - SRT 支持 `[角色] 台词`、`角色：台词` 和 `[角色|emotion=text:生气] 台词`，输出结构化预览
   - 台词输入关闭 ComfyUI 动态提示词解析，JSON 大括号不会在排队时被改写
11. `IndexTTS 2.5 多角色 / SRT 生成 · T8star-Aix`
   - 逐句推理、逐句 AUDIO 列表、合并 AUDIO 和 JSON 报告
   - `shift` 顺延或 `overlay` 时间轴混音；字幕槽位默认使用不丢字的 `pad`，`native/exact` 仅用于允许裁剪的硬槽位
   - 可逐句 ASR 校对并在低于阈值时自动重试，任务报告记录每次 seed、相似度和最终选择
12. `IndexTTS 2.5 人声后处理 · T8star-Aix`
   - 独立处理任意 ComfyUI AUDIO，支持强度混合和目标峰值，不依赖 FFmpeg
13. `IndexTTS 2.5 环境与可选加速 · T8star-Aix`
   - 不加载模型即可检查原生 BF16、FP16、显存、CUDA 工具链、Triton、FlashAttention、DeepSpeed
   - 输出 torch/CUDA Runtime/FlashAttention/Triton/DeepSpeed/Ninja 实际版本、推荐精度、参考编码器位置和安全加速模式
   - 只报告能力，不安装任何附加依赖；“可用”与模型启动后的“实际生效”分开说明
14. `IndexTTS 2.5 时间轴编辑 · T8star-Aix`
   - 接收批量/SRT 脚本和可编辑 JSON，按毫秒修改逐句开始、结束、角色、语言、时长系数与情感覆盖
   - 输出严格校验后的脚本、结构化 JSON 和标准 `IMAGE` 彩色轨道预览
15. `IndexTTS 2.5 ASR 自动校对 · T8star-Aix`
   - 使用可选的本地 OpenAI Whisper 或 faster-whisper 对 AUDIO 识别并与目标文本比较
   - 输出简繁/数字归一化后的 CER/WER、差异明细、词级时间戳、阈值判定、波形对齐图和完整 JSON 报告
16. `IndexTTS 2.5 字幕自动回写 · T8star-Aix`
   - 可保留原 SRT 时间或使用生成音频真实时间轴
   - 可写回原文、全部 ASR 识别结果或仅校对通过的识别结果
17. `IndexTTS 2.5 参考音频质量检测 · T8star-Aix`
   - 检测时长、首尾静音、静音占比、响度、削波、估算信噪比和直流偏移
   - 可在不覆盖原音频的前提下自动裁剪静音，并从超长音频中选取能量最集中的片段
18. `IndexTTS 2.5 显存管理 · T8star-Aix`
   - 查看本扩展 IndexTTS/ASR 模型缓存与 CUDA 显存状态；“全部释放”同时清理 Whisper 缓存
   - 不调用 ComfyUI 全局清理，不会卸载其他节点正在使用的模型
19. `IndexTTS 2.5 audio.cpp 实验生成 · T8star-Aix`
   - 隔离调用可选 `audiocpp_cli` 与 IndexTTS2.5 GGUF，支持 CUDA/CPU/Vulkan/HIP/Metal、五语种、语速与情感控制
   - 不替换默认 Python 推理；CLI 和约 3.5GB 的 Q8 GGUF 均需用户另行下载
20. `IndexTTS 2.5 真实性能基准 · T8star-Aix`
   - 对模型加载器实际生效的模式做预热和 1–5 次正式测量，报告中位/最佳 RTF 与 CUDA 峰值显存
   - 使用相同文本、参考音频和 seed；切换加载器加速模式后重跑即可公平对比
21. `IndexTTS 2.5 检查更新 · T8star-Aix`
   - 手动联网检查官方主分支、官方 Hugging Face 模型和节点版本
   - 只返回 JSON 与摘要，不下载模型、不修改节点、不自动执行

输出固定为 `22050 Hz`、`float32`、`[1,1,T]` 的标准 ComfyUI AUDIO，可直接连接 Save Audio、
音频合并、视频等原生节点。

## 安装

发布到官方 Comfy Registry 后，可在 **ComfyUI Manager** 中搜索 `IndexTTS 2.5 · T8star-Aix`
或节点 ID `indextts25-t8`，点击安装并重启 ComfyUI。模型权重仍需按“模型位置”一节单独下载。

也可以手动安装：

将整个目录放入：

```text
ComfyUI/custom_nodes/comfyui-indextts25-T8
```

也可以直接克隆：

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/comfyui-indextts25-t8.git comfyui-indextts25-T8
```

使用 **ComfyUI 自己的 Python** 安装依赖：

```powershell
python -m pip install -r ComfyUI/custom_nodes/comfyui-indextts25-T8/requirements.txt
```

Windows 便携版通常应从 ComfyUI 目录执行：

```powershell
..\python_embeded\python.exe -m pip install -r custom_nodes\comfyui-indextts25-T8\requirements.txt
```

`requirements.txt` 故意不包含 `torch`、`torchaudio` 和 `torchvision`，避免覆盖 ComfyUI 已安装的
CUDA/PyTorch 组合。若缺少 torchaudio，请安装与现有 torch 完全对应的版本，不要直接执行上游项目的
完整 torch 锁定安装。基础依赖仅列出 ComfyUI 之外的必要运行项；除 `transformers<5` 的上游兼容边界外，
不限制最高版本。

DeepSpeed、FlashAttention、Triton 和 BigVGAN CUDA 编译工具链均为**可选加速**，没有写入
`requirements.txt`。不安装它们时模型加载器和全部普通推理功能仍可使用。

ASR 自动校对也是可选功能。需要使用时，以 ComfyUI 自己的 Python 安装：

```powershell
python -m pip install "openai-whisper>=20250625" "opencc-python-reimplemented>=0.1.7"
```

Windows 便携版：

```powershell
..\python_embeded\python.exe -m pip install "openai-whisper>=20250625" "opencc-python-reimplemented>=0.1.7"
```

如需更快的 CTranslate2 后端，可安装 `faster-whisper>=1.2.0` 和 `opencc-python-reimplemented>=0.1.7`；
也可在节点中明确选择后端。`pyproject.toml` 同时提供 `asr` / `asr-fast` 两组可选依赖供包管理器使用。
首次校对会把所选 Whisper 模型下载到 `ComfyUI/models/TTS/Whisper/`。不安装 Whisper 时，已有的
IndexTTS 生成、时间轴编辑和原文字幕回写仍然正常使用。

当前固定核心已在 Windows、Python 3.10、PyTorch 2.8 环境验证。其他 ComfyUI Python 版本请先运行
环境检查脚本；上游对新版本 Python 的兼容范围可能更窄。

### 依赖版本兼容

`transformers` 已实测兼容 `4.52.1` 和 `4.57.6`（后者对应 `tokenizers 0.22.2`）。依赖范围保持为
`transformers>=4.52.1,<5`；`transformers` 5.x 的生成与缓存 API 变化较大，目前不在支持范围内。

已经使用 `transformers` 4.57.6 的 ComfyUI 无需降级。需要显式升级到已验证组合时，请使用 ComfyUI
自己的 Python：

```powershell
python -m pip install --upgrade "transformers==4.57.6" "tokenizers==0.22.2"
```

Windows 便携版：

```powershell
..\python_embeded\python.exe -m pip install --upgrade "transformers==4.57.6" "tokenizers==0.22.2"
```

升级后重启 ComfyUI。不要单独安装 `transformers` 5.x，也不要因此重装 ComfyUI 的 PyTorch。

中文数字、日期等文本归一化依赖是可选项。在 Windows 上可额外安装 `wetext`；不安装或当前 Python
版本不兼容时，节点会自动使用原文本继续生成，建议把数字写成口语形式。该可选项不会影响 2.5 模型、
语速适配或文本情感功能。

## 模型位置

节点代码和模型权重是两个不同目录，标准位置为：

```text
ComfyUI/
├─ custom_nodes/
│  └─ comfyui-indextts25-T8/       ← 本 GitHub 节点仓库
└─ models/
   └─ TTS/
      └─ IndexTTS-2.5/             ← HF 完整模型仓库的全部文件
         ├─ config.yaml
         ├─ bpe.model
         ├─ gpt.pth
         ├─ codec.pth
         ├─ s2mel.pth
         ├─ feat1.pt
         ├─ feat2.pt
         ├─ wav2vec2bert_stats.pt
         ├─ multilingual_zh_ja_yue_char_del.tiktoken
         ├─ qwen0.6bemo4-merge/
         └─ hf_cache/
            ├─ w2v-bert-2.0/
            ├─ campplus_cn_common.bin
            └─ bigvgan/
```

Windows 完整路径示例：

```text
D:\ComfyUI\models\TTS\IndexTTS-2.5\config.yaml
D:\ComfyUI\models\TTS\IndexTTS-2.5\bpe.model
D:\ComfyUI\models\TTS\IndexTTS-2.5\hf_cache\bigvgan\bigvgan_generator.pt
```

`config.yaml`、`bpe.model` 和 `gpt.pth` 必须直接位于 `IndexTTS-2.5` 目录中。不要出现下面这种多套一层的错误路径：

```text
ComfyUI/models/TTS/IndexTTS-2.5/IndexTTS-2.5-Comfy/config.yaml  ← 错误
```

如果从 HF 网页下载并解压后目录名是 `IndexTTS-2.5-Comfy`，请把它重命名为 `IndexTTS-2.5`，或把里面的全部文件移动到标准目录。手动放置后重启或刷新 ComfyUI。

这是唯一例外：节点所需的代码、清单、脚本、许可证、示例和测试都在本目录；约 7.7 GiB 的完整模型遵循
ComfyUI 模型目录规范，可由多个工作流共享，避免复制。

如果把节点发给其他用户，发送整个 `comfyui-indextts25-T8` 目录即可；对方仍需按本节下载模型，或由你
另行把完整的 `IndexTTS-2.5` 模型目录放到其 `ComfyUI/models/TTS/`。不要只发送单个 `.py` 文件。

### 方法一：在模型加载器中按需下载（推荐）

1. 添加“IndexTTS 2.5 模型加载器”。
2. 开启“缺失时自动下载/修复完整模型”。
3. 阅读许可证与免责声明后，勾选“我已阅读并接受模型许可证”。
4. 运行一次工作流。节点会下载或修复到上面的标准目录，并在完成后继续加载；默认不会自动联网。

完整仓库约 7.7 GiB，请保证磁盘空间和网络稳定。首次下载完成后可关闭这两个选项。

### 方法二：命令行下载

节点已放到 ComfyUI 后，Hugging Face 是默认和推荐下载源：

```powershell
python ComfyUI/custom_nodes/comfyui-indextts25-T8/scripts/download_models.py `
  --source huggingface `
  --accept-license
```

也可使用 Hugging Face CLI 直接下载完整目录：

```powershell
hf download t8star/IndexTTS-2.5-Comfy `
  --local-dir "ComfyUI/models/TTS/IndexTTS-2.5"
```

ModelScope 是兼容下载路径。使用前先安装可选依赖：

```powershell
python -m pip install -r ComfyUI/custom_nodes/comfyui-indextts25-T8/requirements-modelscope.txt
```

```powershell
python ComfyUI/custom_nodes/comfyui-indextts25-T8/scripts/download_models.py `
  --source modelscope `
  --accept-license
```

节点未安装到 `custom_nodes` 时，显式指定 ComfyUI 根目录：

```powershell
python scripts/download_models.py --comfy-root "D:\ComfyUI" --source huggingface --accept-license
```

下载器固定到清单中的完整模型版本，完成后执行全量 SHA-256 校验，并准备 Wav2Vec2-BERT、
CAMPPlus 和 BigVGAN 辅助模型。直接下载官方 `IndexTeam/IndexTTS-2.5` 仓库不会包含必需的
`bpe.model`；请使用上面的完整仓库或本项目下载器。手动安装后重启/刷新 ComfyUI，模型下拉框才会刷新。

## 快速使用

1. 用 ComfyUI 原生 `Load Audio` 载入清晰、单人、无背景音乐的参考音频，建议 3–10 秒。
2. 添加模型加载器并选择 `IndexTTS-2.5`。
3. 添加语音生成节点，连接模型和参考音频，填写文本与语言。
4. 需要多音字或专有词发音时，添加发音控制节点，将其文本输出连接到语音生成节点。
5. 可选连接情感控制和采样设置；长文本建议先经过“分段与停顿预览”。
6. 将生成 AUDIO 连接到 `Save Audio`，也可再连接独立“人声后处理”节点。

### 多角色、批量台词与 SRT

1. 每个角色添加一个“角色音色”节点，连接对应的 `Load Audio`。
2. 需要独立情感时，每个角色分别添加“情感控制”，连接到其“该角色默认情感”输入。
3. 把这些角色连接到“角色音色 / 情感合并”；也可使用同功能的 `Merge Voice Emotions` 节点。
4. 添加“批量台词 / SRT”节点并选择格式。
5. 把模型、角色库、脚本连接到“多角色 / SRT 生成”。

```text
情感控制A ──► 角色音色A ┐
参考音频A ──►            │
                         ├─► 角色音色/情感合并 ─► 多角色/SRT生成
情感控制B ──► 角色音色B │
参考音频B ──►            ┘
```

每条台词只会读取对应角色保存的情感，不会串到其他角色。完整可运行连接见
`23_multi_role_emotions.json`。这里的“合并”表示汇总多个角色配置；如果需要把“悲伤 60% + 愤怒
40%”混成同一个八维情绪，应直接在单个“情感控制”节点中设置这两个维度。

批量文本每行格式如下；语言、时长系数和逐句情感都可省略：

```text
角色A|我先平静地说明情况。|ZH|1.0|text:平静、从容
角色A|你为什么一直在骗我！|ZH|1.0|vector:0,0.8,0,0,0,0,0,0
角色A|这一句恢复角色默认情感。|ZH|1.0
```

也支持 JSON：

```json
[
  {
    "role": "角色A",
    "text": "我先平静地说明情况。",
    "language": "ZH",
    "duration_factor": 1.0,
    "emotion": {"mode": "text", "text": "平静、从容", "strength": 0.75}
  },
  {
    "role": "角色A",
    "text": "你为什么一直在骗我！",
    "language": "ZH",
    "duration_factor": 1.0,
    "emotion": {
      "mode": "vector",
      "vector": [0, 0.8, 0, 0, 0, 0, 0, 0],
      "strength": 0.85
    }
  }
]
```

八维顺序固定为 **喜、怒、哀、惧、厌恶、低落、惊喜、平静**，每项范围 `0–1`。逐句支持
`inherit / speaker / text / vector`：`inherit`（或留空）使用角色音色节点保存的默认情感；`speaker`
强制跟随音色参考；`text` 使用情感描述；`vector` 使用八维向量。完整工作流见
`31_per_line_emotion.json`。

JSON 可直接粘贴或随示例工作流载入。v0.5.1 起该输入明确关闭 ComfyUI 的动态提示词解析；
旧版把 JSON 大括号误当作动态提示词，可能在排队时出现 `Expecting ',' delimiter`。

SRT 示例：

```srt
1
00:00:00,500 --> 00:00:02,600
[角色A|emotion=text:平静、从容] 这是第一条字幕。

2
00:00:02,800 --> 00:00:05,000
[角色A|emotion=vector:0,0.8,0,0,0,0,0,0] 同一个角色现在变得生气。
```

`shift` 会把发生冲突的语音顺延，避免重叠；`overlay` 保留原始 SRT 起点并对重叠部分安全混音。
“适配字幕槽位”默认选择 `pad`：短音频补静音，超长音频完整保留，避免因字幕槽位过短裁掉句尾。
`native` 会直接向长度调节器分配目标帧数，但最后仍会精确补齐或裁剪；`exact` 也会裁剪，只有在
必须严格对齐且已经确认整句能放入槽位时才建议使用。目标时长与自然长度差异过大时仍可能降低自然度。

### 自动分段、停顿与目标秒数

采样设置默认使用 `auto` 分段：英语和西班牙语按 60 Token、阿拉伯语按 80 Token、日语按
100 Token、中文按 120 Token。需要复现实验参数时可切换 `custom`。连接“分段与停顿预览”后，
JSON 会列出模型输入前的 Token 分段，以及每段前后的外加静音。

停顿预设包括 `off / natural / narration / dialogue / custom`。无论选择哪个预设，都能在正文中写：

```text
第一句结束。<pause=0.8>八百毫秒后继续。
也可以写成<pause=500ms>五百毫秒。
```

标点预设会把句子拆成独立语音块，因此比只调整“段间静音”更精确，但推理次数也会增加。
v0.8.1 已回移 GPT 合成提示 KV Cache 根修复；多个停顿语音块和长文本不再仅因该边界问题关闭
`gpt_accel`。采样参数语义不兼容时仍会自动使用普通 GPT 路径。

语音生成的“目标时长（秒）”提供五种模式：

- `off`：只使用原始 `duration_factor`。
- `native`：按文本权重把总秒数分配给长度调节器，一次推理完成；最后精确收尾，可能裁掉超长句尾。
- `natural`：根据首轮实测时长自动计算 0.5–2.0 的新系数并再推理一次，不裁剪。
- `pad`：自然适配后，不足补静音；超长语音完整保留并在报告中标记，是字幕槽位的安全默认值。
- `exact`：自然适配后补静音或强制裁剪到精确采样点；裁剪可能切掉尾音，字幕硬槽位才建议使用。

内置后处理预设为 `voice_clarity / clear_narration / deharsh / warm / normalize`。生成节点可直接选择，
也可以用独立“人声后处理”节点比较原音和处理音；`off` 时不会改变波形。

### CFM 高级参数

采样设置的高级输入直接作用于声谱图 CFM 阶段：

- `diffusion_steps`：默认 25；提高到 40–50 通常更稳，但耗时近似按步数增加。
- `inference_cfg_rate`：默认 0.7；提高可增强参考音色/音高约束，过高可能过度平滑。
- `cfm_temperature`：默认 1.0；降低到 0.8 可减少抖动。

建议一次只改一项并固定生成节点的 `seed` 做 A/B 对比。示例 `19_cfm_advanced.json` 使用
`40 / 0.85 / 0.8`，不是新的强制默认值。

### 可选加速模式

模型加载器默认 `off`，这是零附加依赖、兼容性最高的模式：

- `precision=auto`：原生 BF16 可用时选择 `bfloat16`，否则 CUDA 自动使用 `float16`；CPU 使用 `float32`。
- `reference_device=auto`：显存低于 10GB 时把参考编码器放 CPU；`same` 强制同设备，`cpu` 始终节省显存。
- `reuse_spk_cond_for_emo`：只有未连接独立情感时才复用音色条件，默认关闭；开启后更快但可能轻微改变听感。

- `auto_safe`：仅当本机已有 Ninja 和 CUDA/C++ 编译工具链时启用 BigVGAN CUDA 融合核。
- `bigvgan_cuda`：显式请求 BigVGAN 融合核；首次可能编译，失败自动回退。
- `torch_compile`：需要与当前 PyTorch 匹配的 Triton；首次推理有编译开销。
- `gpt_accel`：需要 FlashAttention 和 Triton。该路径不能完整表达所有 beam/top-p/top-k 参数，节点发现
  不兼容采样组合时会临时使用普通 GPT，避免静默改变结果语义；合成提示 KV Cache 已在内置核心修复。
- `deepspeed`：只在用户显式选择且环境已安装 DeepSpeed 时启用。节点不会替用户安装，也不会把它
  作为必需依赖；不同硬件上可能加速，也可能更慢。

缺依赖或初始化失败时，模型信息会显示 `effective=off` 和回退原因。先用“环境与可选加速”节点检查，
再决定是否维护独立的加速环境。不要为了节点加速覆盖 ComfyUI 的 torch/CUDA 组合。

Windows、Python 3.10、`torch 2.8.0+cu128` 可使用桌面整合包已实测的精确轮子：

```powershell
pip install "triton-windows==3.4.0.post21"
pip install "https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu128torch2.8.0cxx11abiFALSE-cp310-cp310-win_amd64.whl"
pip install "https://github.com/6Morpheus6/deepspeed-windows-wheels/releases/download/v0.17.5/deepspeed-0.17.5%2Be1560d84-2.8torch_cu128-cp310-cp310-win_amd64.whl"
```

这组轮子只适用于上述精确 ABI；其他 Python、torch 或 CUDA 组合必须寻找对应轮子。节点不会自动安装，
也不会把它们加入基础 `requirements.txt`，以免覆盖宿主 ComfyUI 的 PyTorch 环境。

vLLM-Omni 更适合 Linux 服务端吞吐场景，后续会作为隔离 sidecar 评估；当前不会塞入 Windows/ComfyUI
基础环境。官方 TensorRT 后端目前只提供 IndexTTS 2.0 引擎，因此本节点不会虚假宣称 2.5 TensorRT。

超过 15 秒的参考音频会被截取并提示。节点会把参考音频按内容哈希缓存到 ComfyUI 的临时目录，
不会污染 input/output。相同模型的并发推理会串行化，防止上游内部音色/情感缓存互相覆盖。

`duration_factor` 表示目标时长倍率：

- `0.5`：更短、更快
- `1.0`：默认时长
- `2.0`：更长、更慢

它是模型内的长度调节，不是简单的后处理拉伸；仍不承诺逐字或字幕级精确时长。

### ASR 自动校对、字幕回写与时间轴编辑

ASR 校对节点接收标准 ComfyUI `AUDIO` 和目标文本，在本机运行 Whisper。后端可选
`auto / openai_whisper / faster_whisper`，语言可选 `AUTO / ZH / EN / JA / ES / AR`，模型可选
`tiny / base / small / medium / turbo`，设备可选 `auto / CUDA / CPU`。中文和日文按字符错误率 CER，
英文、西语和阿语按词错误率 WER 判定；自动语言会根据文本选择指标。比对前统一 NFKC、大小写、
简繁体、中文数字和标点，报告包含差异明细和词级时间戳。`tiny` 下载快，较大模型通常更准确，
但占用更多磁盘、显存和时间。音频以波形直接送入 Whisper，不依赖系统 FFmpeg。

多角色/SRT 生成节点可在每句生成后自动校对，同时输出 `rewritten_srt` 和 `timeline_report`。字幕时间有：

- `original`：保留原始 SRT 起止时间；批量台词没有原始时段时使用实际时间轴。
- `actual`：使用最终合并音频中的真实开始/结束时间。

字幕文本有：

- `original`：始终使用输入台词。
- `asr_all`：有识别结果时全部替换。
- `asr_passed`：仅相似度达到阈值时替换，其余保留原文。

ASR 属于可选的生成后校对：后端未安装或单句识别失败时，节点会在报告中记录警告、保留原字幕，
并继续输出已经生成的音频，不会因为字幕回写而让整项任务失败。旧工作流若把字幕选项保存成 `0 / 1 / 2`
或因控件顺序变化发生错位，也会自动转换为安全模式。若要手动修复旧节点，可把“回写字幕时间”设为
`actual`，把“回写字幕文本”设为 `asr_passed`；仍显示数字时删除并重新添加该生成节点。

“时间轴编辑”节点的 JSON 数组每行使用毫秒，真实示例如下；行号必须完整且不重复：

```json
[
  {"line": 1, "role": "角色A", "text": "第一句。", "language": "ZH", "duration_factor": 1.0, "start_ms": 500, "end_ms": 2100},
  {"line": 2, "role": "角色B", "text": "Second line.", "language": "EN", "duration_factor": 0.9, "start_ms": 2300, "end_ms": 4500}
]
```

把 `可视化时间轴` 连接到 ComfyUI 原生 `Preview Image` 即可看到每句彩色轨道。把编辑后的
`台词脚本` 连接到“多角色 / SRT 生成”，即可按新时间轴混音；ASR 报告也可单独连接
“字幕自动回写”节点，在不重新推理的情况下尝试不同的时间和文本回写策略。

### 多音字与精确发音

不连接发音控制节点时，也可以在语音生成正文中直接使用官方格式：

```text
小明<要求|YAO4 QIU2>这个题的答案是多少。
他在<银行|YIN2 HANG2>里<行走|XING2 ZOU3>了半天。
He had a <minute|M IH1 . N AH0 T> to check the <minute|M AY0 . N UW1 T> details.
彼は料理が<上手|じょうず>だが、囲碁では<上手|うわて>に負けた。
```

中文必须做到“一个汉字对应一个带声调拼音”。多音字位于连续词语中时请标注完整词语：官方问题
#792 的 `小明<要|YAO4>求…` 可能被上下文词义覆盖，应写成 `小明<要求|YAO4 QIU2>…`。
节点会对单字嵌入连续中文和汉字/音节数量不一致给出警告。

批量规则建议使用发音控制节点。词典每行格式为 `文字|读音|语言`，例如：

```text
银行|YIN2 HANG2|ZH
行长|HANG2 ZHANG3|ZH
Bilibili|B IY1 . L IY1 . B IY1 . L IY1|EN
```

词典保存在工作流 JSON 内，发送工作流时不会丢失。已有手工标注永远优先；词典按照长词优先，
不会改写 `<文字|读音>` 内部。严格校验默认开启，错误会在排队前给出；关闭后无效词条保持原文并
写入报告。该节点不依赖额外 G2P 模型，也不会修改已缓存模型的全局 glossary。

完整示例见 `example_workflows/README.md`，包含 31 组可直接打开的 UI 工作流和 31 组 API prompt：
基础克隆、语速对比、情感参考音频、八维情感、文本情感、随机采样长文本、五语种生成，以及中文
多音字、英文 CMU 音素、日语假名发音控制、多角色、JSON 批量台词、SRT、可选加速诊断、自动分段
预览、显式停顿、原生目标秒数、CFM 高级参数、独立音频后处理、ASR 自动校对、时间轴编辑、字幕
回写、多角色独立情感、参考音频检测、ASR 失败重试、模型回收、audio.cpp 实验后端和低显存 FP16。
使用前把
`voice_reference.wav`（情感音频示例还需 `emotion_reference.wav`）上传到 ComfyUI input。

### 可选 audio.cpp 实验后端

节点只提供安全的无 shell CLI 连接器，不捆绑第三方二进制或 GGUF 模型，也不会改变默认加载器。
从 [audio.cpp 官方发布页](https://github.com/0xShug0/audio.cpp/releases) 下载对应 Windows CLI，模型从
[audio.cpp GGUF 仓库](https://huggingface.co/audio-cpp/audio.cpp-gguf) 的 `IndexTTS2.5-GGUF` 获取。
当前 Q8 文件约 3.5GB。audio.cpp 的文本归一化是独立 C++ 实现，少见日期、单位、网址，以及日语/西语
分词边界可能与官方 Python 路径不同；正式使用前应对五语种、情感、发音标注和语速分别试听对比。

## 环境与模型检查

```powershell
python scripts/check_environment.py
python scripts/check_environment.py "D:\ComfyUI\models\TTS\IndexTTS-2.5"
python scripts/download_models.py --target "D:\ComfyUI\models\TTS\IndexTTS-2.5" --verify-only
```

最后一条命令会读取约 7.7 GiB 文件执行完整哈希校验。

Registry 安全扫描说明与剩余敏感操作的边界见 [`SECURITY_REVIEW.md`](SECURITY_REVIEW.md)。发布工作流
不会再把“上传完成”误报成“Manager 可安装”；只有版本状态成为 `NodeVersionStatusActive` 才通过。

## 固定版本

- IndexTTS 代码：`ee40fa7d6c6b8a2c7f06105f9f1e65775b74868c`
- IndexTTS 2.5 上游模型：`c39ce5ba981572cb187443877ff559dfb246ce63`
- 完整模型仓库：`14166a7401f9f87f53770a1784390e8c0e9da15a`
- 模型清单：`manifests/model_2_5.json`

## 官方项目、模型下载与致谢

- IndexTTS 官方仓库：[index-tts/index-tts](https://github.com/index-tts/index-tts)
- 本节点完整模型（推荐）：[t8star/IndexTTS-2.5-Comfy](https://huggingface.co/t8star/IndexTTS-2.5-Comfy)
- IndexTTS 2.5 模型（ModelScope）：[IndexTeam/IndexTTS-2.5](https://modelscope.cn/models/IndexTeam/IndexTTS-2.5)
- IndexTTS 2.5 模型（Hugging Face）：[IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5)

完整模型仓库仅重新组织运行所需的原始文件，不修改上游权重，并在模型卡中列出每个来源、固定版本和许可证。特别感谢 IndexTTS、Wav2Vec2-BERT、CAMPPlus 与 BigVGAN 的作者开源相关成果。

感谢 IndexTTS 团队开源 IndexTTS 及 IndexTTS 2.5 模型。本项目是在其开源成果基础上开发的第三方
ComfyUI 集成；请支持并关注官方项目。

当前节点已同步上游 2.5 的 QwenEmotion 标签兼容、无效 `use_gpt_latent` 路径移除、
torchaudio 2.9+ WAV 防削波修复。低于 10GB 显存时会自动使用低显存策略：长文本分段，
文本情感分析完成后先释放 QwenEmotion，再执行语音生成。

## 发布维护

推送 GitHub 前，使用仓库脚本自动更新 `pyproject.toml` 的语义化版本：

```powershell
python scripts/bump_version.py patch
```

修复和维护使用 `patch`，向后兼容的新功能使用 `minor`，破坏性变更使用 `major`。仓库级
`AGENTS.md` 已要求后续自动化代理在每次推送前检查并更新版本，避免重复使用 Comfy Registry
中不可变的版本号。

## 许可证与免责声明

分发或使用前必须阅读 `LICENSE`、`LICENSE_ZH.txt`、`DISCLAIMER` 和 `THIRD_PARTY_NOTICES.md`。
模型许可证要求下游保留相关版权和许可信息，并对衍生品作出非背书/不担保声明。

本节点为第三方集成，不是哔哩哔哩或 IndexTTS 原始权利人的官方产品；原始权利人不对本衍生品背书、
不担保、不承担责任。
