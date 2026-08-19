# comfyui-indextts25-T8

IndexTTS 2.5 的 ComfyUI V3 原生节点集成。节点菜单位于：

`T8star-Aix / Audio / IndexTTS 2.5`

作者标识：**B 站：T8star-Aix**。

本目录固定使用 IndexTTS 2.5 推理核心和正式 2.5 模型清单，不会回退或误载 IndexTTS 2.0。

## 已实现节点

1. `IndexTTS 2.5 模型加载器 · T8star-Aix`
   - 扫描标准模型目录和 `extra_model_paths.yaml` 中的 `TTS` 路径
   - 正式模型文件大小校验；可选完整 SHA-256 校验
   - `auto / CUDA / CPU` 设备和 `auto / bfloat16 / float32` 精度
   - 全局惰性缓存、同模型线程锁、可选生成后释放
2. `IndexTTS 2.5 情感控制 · T8star-Aix`
   - 跟随音色参考
   - 独立情感参考音频
   - 八维情感向量（高兴、愤怒、悲伤、恐惧、厌恶、低落、惊讶、自然）
   - 文本情感描述；Qwen 情感模型按需加载到同一推理设备
3. `IndexTTS 2.5 采样设置 · T8star-Aix`
   - 稳定默认值、随机采样、beam、temperature、top-p/top-k
   - 长文本分段、段间停顿、最大语音 token、文本归一化
4. `IndexTTS 2.5 发音控制 · T8star-Aix`
   - 官方 `<文字|读音>` 标注格式
   - 中文数字声调拼音、英文 CMU 音素、日语假名校验
   - 工作流内嵌发音词典、长词优先、手工标注优先
   - 输出处理后文本和完整替换/校验报告，不修改模型全局状态
5. `IndexTTS 2.5 语音生成 · T8star-Aix`
   - 标准 ComfyUI `AUDIO` 输入/输出
   - 中、英、日、西、阿五种语言入口
   - 音色克隆、seed、官方 `duration_factor=0.5~2.0` 语速/时长适配
6. `IndexTTS 2.5 角色音色 · T8star-Aix`
   - 将角色名、标准 AUDIO、默认语言和可选情感封装成工作流内音色
7. `IndexTTS 2.5 角色音色库 · T8star-Aix`
   - 自动增长输入，可连接 1–16 个角色；重复角色名会在排队前报错
8. `IndexTTS 2.5 批量台词 / SRT · T8star-Aix`
   - 解析 `角色|台词|语言|时长系数`、JSON 数组和标准 SRT
   - SRT 支持 `[角色] 台词` 与 `角色：台词`，输出结构化预览
   - 台词输入关闭 ComfyUI 动态提示词解析，JSON 大括号不会在排队时被改写
9. `IndexTTS 2.5 多角色 / SRT 生成 · T8star-Aix`
   - 逐句推理、逐句 AUDIO 列表、合并 AUDIO 和 JSON 报告
   - `shift` 顺延或 `overlay` 时间轴混音；可选二次推理贴合字幕槽位
10. `IndexTTS 2.5 环境与可选加速 · T8star-Aix`
   - 不加载模型即可检查 BF16、CUDA 工具链、Triton、FlashAttention、DeepSpeed
   - 只报告能力，不安装任何附加依赖

输出固定为 `22050 Hz`、`float32`、`[1,1,T]` 的标准 ComfyUI AUDIO，可直接连接 Save Audio、
音频合并、视频等原生节点。

## 安装

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
5. 可选连接情感控制和采样设置。
6. 将生成 AUDIO 连接到 `Save Audio` 或 `Preview Audio`。

### 多角色、批量台词与 SRT

1. 每个角色添加一个“角色音色”节点，连接对应的 `Load Audio`。
2. 把这些音色连接到“角色音色库”的自动增长输入。
3. 添加“批量台词 / SRT”节点并选择格式。
4. 把模型、角色库、脚本连接到“多角色 / SRT 生成”。

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
“适配字幕槽位”会根据第一次生成的时长计算 0.5–2.0 范围内的新 `duration_factor`，最多再推理一次。
它只能尽量贴合，报告中的 `overrun_ms` 才是最终超时依据，不承诺逐帧精确同步。

### 可选加速模式

模型加载器默认 `off`，这是零附加依赖、兼容性最高的模式：

- `auto_safe`：仅当本机已有 Ninja 和 CUDA/C++ 编译工具链时启用 BigVGAN CUDA 融合核。
- `bigvgan_cuda`：显式请求 BigVGAN 融合核；首次可能编译，失败自动回退。
- `torch_compile`：需要与当前 PyTorch 匹配的 Triton；首次推理有编译开销。
- `gpt_accel`：需要 FlashAttention 和 Triton。该路径不能完整表达所有 beam/top-p/top-k 参数，节点发现
  不兼容采样组合时会临时使用普通 GPT，避免静默改变结果语义。
- `deepspeed`：只在用户显式选择且环境已安装 DeepSpeed 时启用。节点不会替用户安装，也不会把它
  作为必需依赖；不同硬件上可能加速，也可能更慢。

缺依赖或初始化失败时，模型信息会显示 `effective=off` 和回退原因。先用“环境与可选加速”节点检查，
再决定是否维护独立的加速环境。不要为了节点加速覆盖 ComfyUI 的 torch/CUDA 组合。

vLLM-Omni 更适合 Linux 服务端吞吐场景，后续会作为隔离 sidecar 评估；当前不会塞入 Windows/ComfyUI
基础环境。官方 TensorRT 后端目前只提供 IndexTTS 2.0 引擎，因此本节点不会虚假宣称 2.5 TensorRT。

超过 15 秒的参考音频会被截取并提示。节点会把参考音频按内容哈希缓存到 ComfyUI 的临时目录，
不会污染 input/output。相同模型的并发推理会串行化，防止上游内部音色/情感缓存互相覆盖。

`duration_factor` 表示目标时长倍率：

- `0.5`：更短、更快
- `1.0`：默认时长
- `2.0`：更长、更慢

它是模型内的长度调节，不是简单的后处理拉伸；仍不承诺逐字或字幕级精确时长。

### 多音字与精确发音

不连接发音控制节点时，也可以在语音生成正文中直接使用官方格式：

```text
他在银<行|XING2>里<行|HANG2>走了半天。
He had a <minute|M IH1 . N AH0 T> to check the <minute|M AY0 . N UW1 T> details.
彼は料理が<上手|じょうず>だが、囲碁では<上手|うわて>に負けた。
```

批量规则建议使用发音控制节点。词典每行格式为 `文字|读音|语言`，例如：

```text
银行|YIN2 HANG2|ZH
行长|HANG2 ZHANG3|ZH
Bilibili|B IY1 . L IY1 . B IY1 . L IY1|EN
```

词典保存在工作流 JSON 内，发送工作流时不会丢失。已有手工标注永远优先；词典按照长词优先，
不会改写 `<文字|读音>` 内部。严格校验默认开启，错误会在排队前给出；关闭后无效词条保持原文并
写入报告。该节点不依赖额外 G2P 模型，也不会修改已缓存模型的全局 glossary。

完整示例见 `example_workflows/README.md`，包含 14 组可直接打开的 UI 工作流和 14 组 API prompt：
基础克隆、语速对比、情感参考音频、八维情感、文本情感、随机采样长文本、五语种生成，以及中文
多音字、英文 CMU 音素、日语假名发音控制、多角色、JSON 批量台词、SRT 和可选加速诊断。使用前把
`voice_reference.wav`（情感音频示例还需 `emotion_reference.wav`）上传到 ComfyUI input。

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

## 许可证与免责声明

分发或使用前必须阅读 `LICENSE`、`LICENSE_ZH.txt`、`DISCLAIMER` 和 `THIRD_PARTY_NOTICES.md`。
模型许可证要求下游保留相关版权和许可信息，并对衍生品作出非背书/不担保声明。

本节点为第三方集成，不是哔哩哔哩或 IndexTTS 原始权利人的官方产品；原始权利人不对本衍生品背书、
不担保、不承担责任。
