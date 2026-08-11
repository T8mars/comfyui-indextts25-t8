# 第三方组件与模型声明

本节点包内置了 IndexTTS 2.5 推理核心的固定副本，以避免运行时错误引用 IndexTTS 2.0。

- 上游代码仓库：`index-tts/index-tts`
- 固定代码提交：`56eead7eb0888ecac6abbf9d777c27f798a2c730`
- 正式模型仓库：`IndexTeam/IndexTTS-2.5`
- 固定模型版本：`ba2480d9f7f629eb18f6acaebb357679d9ba88a4`
- 上游许可证及限制：见本目录 `LICENSE`、`LICENSE_ZH.txt` 和 `DISCLAIMER`

模型权重不包含在本节点包内。下载脚本会要求用户显式传入 `--accept-license`，并在下载后依据
`manifests/model_2_5.json` 对正式模型文件执行 SHA-256 校验。

## 衍生品声明

`comfyui-indextts25-T8` 是第三方 ComfyUI 集成，并非 IndexTTS 原始权利人或哔哩哔哩官方产品。

该衍生品对原模型所作的任何改动与原模型原始权利人无关，原始权利人对该衍生品不背书、
不担保、不承担责任。`T8star-Aix` 标识仅用于说明本节点集成的作者/发布者，不表示上游官方背书。

