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
4. `IndexTTS 2.5 语音生成 · T8star-Aix`
   - 标准 ComfyUI `AUDIO` 输入/输出
   - 中、英、日、西、阿五种语言入口
   - 音色克隆、seed、官方 `duration_factor=0.5~2.0` 语速/时长适配

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

当前固定核心已在 Windows、Python 3.10、PyTorch 2.8 环境验证。其他 ComfyUI Python 版本请先运行
环境检查脚本；上游对新版本 Python 的兼容范围可能更窄。

Transformers 已实测兼容 `4.52.1` 和 `4.57.6`（后者对应 `tokenizers 0.22.2`）。依赖范围保持为
`transformers>=4.52.1,<5`；Transformers 5.x 的生成与缓存 API 变化较大，目前不在支持范围内。

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
4. 可选连接情感控制和采样设置。
5. 将生成 AUDIO 连接到 `Save Audio` 或 `Preview Audio`。

超过 15 秒的参考音频会被截取并提示。节点会把参考音频按内容哈希缓存到 ComfyUI 的临时目录，
不会污染 input/output。相同模型的并发推理会串行化，防止上游内部音色/情感缓存互相覆盖。

`duration_factor` 表示目标时长倍率：

- `0.5`：更短、更快
- `1.0`：默认时长
- `2.0`：更长、更慢

它是模型内的长度调节，不是简单的后处理拉伸；仍不承诺逐字或字幕级精确时长。

完整示例见 `example_workflows/README.md`，包含 7 组可直接打开的 UI 工作流和 7 组 API prompt：
基础克隆、语速对比、情感参考音频、八维情感、文本情感、随机采样长文本和五语种生成。使用前把
`voice_reference.wav`（情感音频示例还需 `emotion_reference.wav`）上传到 ComfyUI input。

## 环境与模型检查

```powershell
python scripts/check_environment.py
python scripts/check_environment.py "D:\ComfyUI\models\TTS\IndexTTS-2.5"
python scripts/download_models.py --target "D:\ComfyUI\models\TTS\IndexTTS-2.5" --verify-only
```

最后一条命令会读取约 5GB 文件执行完整哈希校验。

## 固定版本

- IndexTTS 代码：`56eead7eb0888ecac6abbf9d777c27f798a2c730`
- IndexTTS 2.5 模型：`ba2480d9f7f629eb18f6acaebb357679d9ba88a4`
- 模型清单：`manifests/model_2_5.json`

## 官方项目、模型下载与致谢

- IndexTTS 官方仓库：[index-tts/index-tts](https://github.com/index-tts/index-tts)
- IndexTTS 2.5 模型（ModelScope）：[IndexTeam/IndexTTS-2.5](https://modelscope.cn/models/IndexTeam/IndexTTS-2.5)
- IndexTTS 2.5 模型（Hugging Face）：[IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5)

感谢 IndexTTS 团队开源 IndexTTS 及 IndexTTS 2.5 模型。本项目是在其开源成果基础上开发的第三方
ComfyUI 集成；请支持并关注官方项目。

## 许可证与免责声明

分发或使用前必须阅读 `LICENSE`、`LICENSE_ZH.txt`、`DISCLAIMER` 和 `THIRD_PARTY_NOTICES.md`。
模型许可证要求下游保留相关版权和许可信息，并对衍生品作出非背书/不担保声明。

本节点为第三方集成，不是哔哩哔哩或 IndexTTS 原始权利人的官方产品；原始权利人不对本衍生品背书、
不担保、不承担责任。
