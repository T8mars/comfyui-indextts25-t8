# comfyui-indextts25-T8

**简体中文** | [English](README_EN.md)

IndexTTS 2.5 的 ComfyUI V3 原生节点集成。节点菜单位于：

`T8star-Aix / Audio / IndexTTS 2.5`

作者标识：**B 站：T8star-Aix**。

## 作者与资源链接

- B 站：[T8star-Aix](https://space.bilibili.com/385085361)
- YouTube：[T8star-Aix](https://www.youtube.com/@T8star-Aix/)
- Hugging Face：[t8star](https://huggingface.co/t8star)
- API 注册（推广链接）：[api.seedance.nz](https://api.seedance.nz/sign-up?aff=5f4w)
- 在线 AI 应用（RunningHub）：[T8star-Aix 用户主页](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- ComfyUI 整合包：[夸克网盘下载](https://pan.quark.cn/s/264edb7e36bd)
- IndexTTS 2.5 模型网盘：[夸克网盘下载](https://pan.quark.cn/s/c9c267081fbf)

本目录固定使用 IndexTTS 2.5 推理核心和正式 2.5 模型清单，不会回退或误载 IndexTTS 2.0。

当前版本基线：**ComfyUI Node 0.11.5 · Desktop 0.11.5 · Core `ee40fa7d` · Model `c39ce5ba`**。
Desktop 与 Node 是两个独立发行物，因此各自使用独立版本号；Core/Model 是固定的官方代码和权重 revision。

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
   - `auto / CUDA / CPU` 设备和 `auto / bfloat16 / float32` 精度
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
   - 音色克隆、seed、官方 `duration_factor=0.5~2.0` 语速/时长适配
   - 目标秒数支持原生长度调节器单次适配，以及自然二次适配、补静音、强制精确兼容模式
   - 可选人声清晰、清晰旁白、去刺耳、温暖、峰值归一化后处理
   - 可选本地 ASR 自动质检；失败后更换 seed 重试并保留相似度最高的结果
7. `IndexTTS 2.5 角色音色 · T8star-Aix`
   - 将角色名、标准 AUDIO、默认语言和可选情感封装成工作流内音色
8. `IndexTTS 2.5 角色音色 / 情感合并 · T8star-Aix`
   - 自动增长输入，可汇总 1–16 个角色各自的音色与独立情感；重复角色名会在排队前报错
9. `IndexTTS 2.5 Merge Voice Emotions · T8star-Aix`
   - 对应社区常用的 `Merge Voice Emotions` 搜索名称，输出与“角色音色 / 情感合并”完全一致
   - 合并的是角色配置列表，不会把多个角色的八维情感数值混成一个新情绪
10. `IndexTTS 2.5 批量台词 / SRT · T8star-Aix`
   - 解析 `角色|台词|语言|时长系数`、JSON 数组和标准 SRT
   - 普通批量台词中的 `<文字|读音>` 会作为正文保留；复杂正文也可使用 JSON 格式
   - SRT 支持 `[角色] 台词` 与 `角色：台词`，输出结构化预览
   - 台词输入关闭 ComfyUI 动态提示词解析，JSON 大括号不会在排队时被改写
11. `IndexTTS 2.5 多角色 / SRT 生成 · T8star-Aix`
   - 逐句推理、逐句 AUDIO 列表、合并 AUDIO 和 JSON 报告
   - `shift` 顺延或 `overlay` 时间轴混音；字幕槽位默认使用原生单次适配，也可选择旧版二次推理兼容模式
   - 可逐句 ASR 校对并在低于阈值时自动重试，任务报告记录每次 seed、相似度和最终选择
12. `IndexTTS 2.5 人声后处理 · T8star-Aix`
   - 独立处理任意 ComfyUI AUDIO，支持强度混合和目标峰值，不依赖 FFmpeg
13. `IndexTTS 2.5 环境与可选加速 · T8star-Aix`
   - 不加载模型即可检查 BF16、CUDA 工具链、Triton、FlashAttention、DeepSpeed
   - 只报告能力，不安装任何附加依赖
14. `IndexTTS 2.5 时间轴编辑 · T8star-Aix`
   - 接收批量/SRT 脚本和可编辑 JSON，按毫秒修改逐句开始、结束、角色、语言与时长系数
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

权重不放在 custom node 目录，标准位置为：

```text
ComfyUI/models/TTS/IndexTTS-2.5/
```

这是唯一例外：节点所需的代码、清单、脚本、许可证、示例和测试都在本目录；约 5GB 的模型权重遵循
ComfyUI 模型目录规范，可由多个工作流共享，避免复制。

如果把节点发给其他用户，发送整个 `comfyui-indextts25-T8` 目录即可；对方仍需按本节下载模型，或由你
另行把完整的 `IndexTTS-2.5` 模型目录放到其 `ComfyUI/models/TTS/`。不要只发送单个 `.py` 文件。

在节点已放到 ComfyUI 后执行：

使用 ModelScope 前先安装可选下载依赖：

```powershell
python -m pip install -r ComfyUI/custom_nodes/comfyui-indextts25-T8/requirements-modelscope.txt
```

```powershell
python ComfyUI/custom_nodes/comfyui-indextts25-T8/scripts/download_models.py `
  --source modelscope `
  --accept-license
```

Hugging Face：

```powershell
python ComfyUI/custom_nodes/comfyui-indextts25-T8/scripts/download_models.py `
  --source huggingface `
  --accept-license
```

节点未安装到 `custom_nodes` 时，显式指定 ComfyUI 根目录：

```powershell
python scripts/download_models.py --comfy-root "D:\ComfyUI" --source modelscope --accept-license
```

下载器固定到清单中的正式模型版本，完成后执行全量 SHA-256 校验，并准备 Wav2Vec2-BERT、
MaskGCT、CAMPPlus 和 BigVGAN 辅助模型。下载结束后重启 ComfyUI，模型下拉框才会刷新。

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

批量文本每行格式如下；语言和时长系数可省略：

```text
角色A|你终于来了。|ZH|1.0
角色B|好，我们开始吧。|ZH|0.9
```

也支持 JSON：

```json
[
  {"role": "角色A", "text": "第一句。", "language": "ZH", "duration_factor": 1.0},
  {"role": "角色B", "text": "Second line.", "language": "EN", "duration_factor": 0.9}
]
```

JSON 可直接粘贴或随示例工作流载入。v0.5.1 起该输入明确关闭 ComfyUI 的动态提示词解析；
旧版把 JSON 大括号误当作动态提示词，可能在排队时出现 `Expecting ',' delimiter`。

SRT 示例：

```srt
1
00:00:00,500 --> 00:00:02,600
[角色A] 这是第一条字幕。

2
00:00:02,800 --> 00:00:05,000
角色B：这是第二条字幕。
```

`shift` 会把发生冲突的语音顺延，避免重叠；`overlay` 保留原始 SRT 起点并对重叠部分安全混音。
“适配字幕槽位”默认选择 `native`：直接向长度调节器分配目标帧数，一次推理完成，并在最后做
采样点级收尾。连接外部旧版推理核心且没有 `target_duration` 时，节点会自动回退到原有的
`duration_factor` 二次推理方案，并在生成信息中说明。目标时长与自然语速差异过大时仍可能降低自然度。

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
- `native`：按文本权重把总秒数分配给长度调节器，一次推理完成；包含外加停顿，推荐优先测试。
- `natural`：根据首轮实测时长自动计算 0.5–2.0 的新系数并再推理一次，不裁剪。
- `pad`：自然适配后，不足补静音；超长语音完整保留并在报告中标记。
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

完整示例见 `example_workflows/README.md`，包含 27 组可直接打开的 UI 工作流和 27 组 API prompt：
基础克隆、语速对比、情感参考音频、八维情感、文本情感、随机采样长文本、五语种生成，以及中文
多音字、英文 CMU 音素、日语假名发音控制、多角色、JSON 批量台词、SRT、可选加速诊断、自动分段
预览、显式停顿、原生目标秒数、CFM 高级参数、独立音频后处理、ASR 自动校对、时间轴编辑、字幕
回写、多角色独立情感、参考音频检测、ASR 失败重试、模型回收和 audio.cpp 实验后端。使用前把
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

最后一条命令会读取约 5GB 文件执行完整哈希校验。

## 固定版本

- IndexTTS 代码：`ee40fa7d6c6b8a2c7f06105f9f1e65775b74868c`
- IndexTTS 2.5 模型：`c39ce5ba981572cb187443877ff559dfb246ce63`
- 模型清单：`manifests/model_2_5.json`

## 官方项目、模型下载与致谢

- IndexTTS 官方仓库：[index-tts/index-tts](https://github.com/index-tts/index-tts)
- IndexTTS 2.5 模型（ModelScope）：[IndexTeam/IndexTTS-2.5](https://modelscope.cn/models/IndexTeam/IndexTTS-2.5)
- IndexTTS 2.5 模型（Hugging Face）：[IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5)

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
