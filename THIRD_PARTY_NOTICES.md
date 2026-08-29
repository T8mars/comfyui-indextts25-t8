# 第三方组件与模型声明

本节点包内置了 IndexTTS 2.5 推理核心的固定副本，以避免运行时错误引用 IndexTTS 2.0。

- 上游代码仓库：`index-tts/index-tts`
- 固定代码提交：`ee40fa7d6c6b8a2c7f06105f9f1e65775b74868c`
- 本节点完整模型仓库：`t8star/IndexTTS-2.5-Comfy`
- 固定完整模型版本：`14166a7401f9f87f53770a1784390e8c0e9da15a`
- 上游正式模型仓库：`IndexTeam/IndexTTS-2.5`
- 固定上游模型版本：`c39ce5ba981572cb187443877ff559dfb246ce63`
- 必需分词器 `bpe.model`：`IndexTeam/IndexTTS-2`，固定版本 `740dcaff396282ffb241903d150ac011cd4b1ede`
- 辅助模型：`facebook/w2v-bert-2.0`、`funasr/campplus`、`nvidia/bigvgan_v2_22khz_80band_256x`；来源、固定版本和哈希见模型清单及完整模型仓库的模型卡
- 上游许可证及限制：见本目录 `LICENSE`、`LICENSE_ZH.txt` 和 `DISCLAIMER`

模型权重不包含在本节点包内。下载脚本和模型加载器的按需下载均要求用户显式接受许可证，并在下载后依据
`manifests/model_2_5.json` 对完整模型文件执行 SHA-256 校验。完整模型仓库只重新组织推理必需的原始文件，
不修改上游模型权重。

## 随包第三方源码许可证

节点包中的 BigVGAN 推理源码沿用 NVIDIA BigVGAN 及其引用项目的开源代码。对应的 MIT、
Apache-2.0、BSD-3-Clause 等完整许可证文本已按 NVIDIA BigVGAN 上游提交
`7d2b454564a6c7d014227f635b7423881f14bdac` 原样保存在 `incl_licenses/LICENSE_1` 至
`incl_licenses/LICENSE_8`，并会随 Registry 安装包一同发布。这些第三方许可证与本项目根目录的
IndexTTS 模型使用许可证同时适用；第三方代码的权利仍归各自作者所有。

## 可选 audio.cpp 实验连接器

本节点仅提供对 [audio.cpp](https://github.com/0xShug0/audio.cpp) 命令行接口的隔离调用，
不复制、修改或分发其源码、可执行文件与 GGUF 权重。用户需自行下载第三方 CLI，并根据
[audio.cpp GGUF 仓库](https://huggingface.co/audio-cpp/audio.cpp-gguf) 所列的原模型许可证使用
`IndexTTS2.5-GGUF`。此实验连接器不会替换本项目默认的官方 Python/IndexTTS 2.5 推理路径。

## 衍生品声明

`comfyui-indextts25-T8` 是第三方 ComfyUI 集成，并非 IndexTTS 原始权利人或哔哩哔哩官方产品。

该衍生品对原模型所作的任何改动与原模型原始权利人无关，原始权利人对该衍生品不背书、
不担保、不承担责任。`T8star-Aix` 标识仅用于说明本节点集成的作者/发布者，不表示上游官方背书。
