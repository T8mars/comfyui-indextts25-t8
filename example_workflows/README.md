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
| `06_random_sampling_long_text.json` | 长英文分段、防截断自动重试与固定 seed 随机采样 | `voice_reference.wav` |
| `07_multilingual_generation.json` | 中文、英语、日语、西班牙语、阿拉伯语 | `voice_reference.wav` |
| `08_chinese_pronunciation.json` | 中文多音字词典、长词优先和手工标注优先 | `voice_reference.wav` |
| `09_english_cmu_pronunciation.json` | 英文同形异音与 CMU 音素校验 | `voice_reference.wav` |
| `10_japanese_kana_pronunciation.json` | 日语假名发音控制 | `voice_reference.wav` |
| `11_multi_role_dialogue.json` | 两个角色音色与多轮对话 | `role_a.wav`、`role_b.wav` |
| `12_batch_dialogue_json.json` | JSON 批量台词、逐句与合并 AUDIO | `role_a.wav`、`role_b.wav` |
| `13_srt_multi_role.json` | SRT 角色标记、长西语台词保护、时间轴和槽位适配 | `role_a.wav`、`role_b.wav` |
| `14_optional_acceleration.json` | `auto_safe` 加速回退与环境诊断 | `voice_reference.wav` |
| `15_auto_segment_preview.json` | 英文 60 Token 自动分段、预览 JSON 与 GPT 加速风险报告 | `voice_reference.wav` |
| `16_pause_control.json` | 旁白标点停顿和 `<pause=0.8>` 显式停顿 | `voice_reference.wav` |
| `17_target_duration.json` | 原生长度调节器单次适配到目标 5 秒 | `voice_reference.wav` |
| `18_audio_postprocess.json` | 独立“清晰旁白”人声后处理节点 | `voice_reference.wav` |
| `19_cfm_advanced.json` | CFM 扩散步数、CFG 引导强度和温度调节 | `voice_reference.wav` |
| `20_asr_proofread.json` | 双 Whisper 后端、CER/WER、差异和词级时间戳校对 | `voice_reference.wav` |
| `21_timeline_editor.json` | 以毫秒编辑起止时间，并用 Preview Image 显示彩色轨道 | `role_a.wav`、`role_b.wav` |
| `22_subtitle_rewrite.json` | 根据生成报告回写实际时间轴和校对通过的 ASR 文本 | 无 |
| `23_multi_role_emotions.json` | 两个角色分别使用独立八维情感，并通过 `Merge Voice Emotions` 汇总 | `role_a.wav`、`role_b.wav` |
| `24_reference_quality.json` | 参考音频质量评分、自动裁剪和波形预览 | `voice_reference.wav` |
| `25_quality_retry.json` | 生成并保留 3 个候选；ASR + 波形质量或纯波形质量自动选优 | `voice_reference.wav` |
| `26_memory_control.json` | 连续生成自动重载、显存/模型状态和参考条件缓存统计 | `voice_reference.wav` |
| `27_audiocpp_experimental.json` | 隔离的 audio.cpp IndexTTS2.5 GGUF 实验后端 | `voice_reference.wav` |
| `28_low_vram_fp16.json` | 低显存 FP16、CPU 参考编码器与快速默认情感 | `voice_reference.wav` |
| `29_runtime_benchmark.json` | 对当前模型加载器实际生效模式测量中位/最佳 RTF 与峰值显存 | `voice_reference.wav` |
| `30_update_check.json` | 手动检查官方代码、官方模型和节点版本，只输出报告 | 无 |
| `31_per_line_emotion.json` | 同一个角色逐句切换文本情感、八维向量与角色默认情感 | `role_a.wav`、`role_b.wav` |
| `32_context_emotion_suggestions.json` | 结合前后文为每句建议八维情感；仅预览和编辑，不自动生成音频 | 无 |
| `33_saved_voice_library.json` | 读取 Desktop 导出的 `.t8voice.zip`，无需重复上传参考音频 | `.t8voice.zip` 音色包 |

## 使用方法

1. 把所需音频上传到 ComfyUI 的 `input` 目录，并保持上表中的文件名；也可以载入工作流后重新选择文件。
2. 确保正式模型位于 `ComfyUI/models/TTS/IndexTTS-2.5/`。
3. 从 `ui/` 打开工作流并排队运行。
4. API 用户可直接读取 `api/` 下的同名文件提交；这些文件使用当前 ComfyUI V3 动态输入要求的
   `mode.xxx` 扁平路径格式。

角色音色库同样使用 V3 自动增长扁平路径 `voices.voice_0`、`voices.voice_1`。SRT 示例默认使用
`pad` 安全适配字幕槽位：短音频补静音、超长音频保留；`native/exact` 才会在最终输出裁剪到精确长度。

示例 23 中，每个“情感控制”先连接到对应的“角色音色”，再由 `Merge Voice Emotions` 汇总角色。
该节点合并的是角色配置列表，不是把两个角色的八维情绪数值平均到一起。
示例 31 则只使用同一个 `角色A` 音色，通过每条 JSON 台词里的 `emotion` 字段逐句覆盖情感；省略
`emotion` 的台词会自动回到角色音色节点保存的默认情感。

示例 32 是安全的“两阶段”流程：先运行本地 QwenEmotion 上下文分析，建议会写入新的台词脚本并送进
时间轴预览；工作流故意不连接生成节点，因此不会在用户确认前合成音频。确认建议后，可在时间轴编辑节点
粘贴并修改报告中的 `lines`，再把它的“编辑后的台词脚本”连接到示例 11 或 13 的多角色生成节点。
`每侧上下文台词数=2` 表示分析目标句时同时参考前后各两句，但提示词会明确区分不同角色。

示例 33 使用桌面端“角色音色库 → 导出音色包”生成的 `.t8voice.zip`。把文件放到
`ComfyUI/models/TTS/IndexTTS-2.5/voices/`，刷新浏览器后从“已保存音色”下拉框选择角色。音色包会携带
该角色的默认语言、情感、标签和质量信息；工作流中不需要 `Load Audio`。

示例 20 需要可选的 `openai-whisper`，首次运行会从网络下载所选模型到
`ComfyUI/models/TTS/Whisper/`。示例 21 的 `start_ms / end_ms` 单位均为毫秒；示例 22 内置的是可直接
运行的真实报告结构示例，用于演示无需重新生成语音的字幕回写。

示例 25 安装 Whisper 时会结合台词相似度和波形质量选优；未安装时仍会保留全部候选并按波形技术指标选优。
示例 27 不使用默认 Python 模型加载器。请先从 audio.cpp 官方发布页和 GGUF 仓库手动下载组件，再填写
`audiocpp_cli` 与 `IndexTTS2.5-GGUF` 的绝对路径；节点不会联网安装或更新这些组件。

示例 28 面向旧显卡和 10GB 以下显存：`float16` 可替代不受原生支持的 `bfloat16`，参考编码器放到
CPU 可减少常驻显存；“快速默认情感”只在未提供独立情感时复用音色条件，速度更快但可能轻微改变听感。
若显卡原生支持 BF16，应优先把精度改回 `auto`。

示例 29 默认先预热一次，再正式测量两次；要比较加速，请只修改模型加载器的 `acceleration_mode`，保持
参考音频、文本、seed、精度和采样设置不变。示例 30 会访问 GitHub 与 Hugging Face，但不会下载或修改文件。

普通批量格式中的正文可以直接包含 `<文字|读音>`，例如
`旁白|小明<要求|YAO4 QIU2>这个题。|ZH|1.0`；解析器不会把标注中的竖线当成字段分隔符。
“显存管理”的“全部释放”会同时清理本扩展加载的 IndexTTS 与可选 Whisper 模型。
示例 26 默认使用 `reference_cache_status`，可查看条目、容量、命中/未命中与命中率。需要手动清理时改为
`clear_reference_cache`；它只删除 `ComfyUI/models/TTS/IndexTTS-2.5/reference_condition_cache/` 下本节点生成的
`safetensors` 条目，不删除模型、角色音频或其他 ComfyUI 缓存。

重新生成全部示例：

```powershell
python scripts/build_example_workflows.py
```

示例生成器会同时检查节点 ID、连接 ID、输入槽位、输出槽位、控件顺序/类型以及 API 节点引用。
