# ComfyUI 示例工作流

本目录保存 `comfyui-indextts25-T8` 的完整示例，每个用法都有两种格式：

- `ui/`：在 ComfyUI 菜单中使用“工作流 → 打开”直接载入。
- `api/`：发送给 ComfyUI `/prompt` 接口的 API prompt JSON。

## 示例清单

| 文件 | 功能 | 所需输入音频 |
| --- | --- | --- |
| `01_basic_voice_clone.json` | 最小音色克隆；情感和采样均使用节点默认值 | `voice_reference.wav` |
| `02_speed_comparison.json` | 官方 `duration_factor` 0.7、1.0、1.3 对比 | `voice_reference.wav` |
| `03_emotion_reference_audio.json` | 音色与情感分别使用独立参考音频 | `voice_reference.wav`、`emotion_reference.wav` |
| `04_emotion_vector.json` | 八维情感向量控制 | `voice_reference.wav` |
| `05_emotion_text.json` | Qwen 文本情感描述控制 | `voice_reference.wav` |
| `06_random_sampling_long_text.json` | 固定 seed 的随机采样、长文本分段和段间静音 | `voice_reference.wav` |
| `07_multilingual_generation.json` | 中文、英语、日语、西班牙语、阿拉伯语 | `voice_reference.wav` |

## 使用方法

1. 把所需音频上传到 ComfyUI 的 `input` 目录，并保持上表中的文件名；也可以载入工作流后重新选择文件。
2. 确保正式模型位于 `ComfyUI/models/TTS/IndexTTS-2.5/`。
3. 从 `ui/` 打开工作流并排队运行。
4. API 用户可直接读取 `api/` 下的同名文件提交；这些文件使用当前 ComfyUI V3 动态输入要求的
   `mode.xxx` 扁平路径格式。

重新生成全部示例：

```powershell
python scripts/build_example_workflows.py
```

示例生成器会同时检查节点 ID、连接 ID、输入槽位、输出槽位以及 API 节点引用。
