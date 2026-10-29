所属项目基线：**V_0**；整体原理和操作见 `../Codex-report-news/News_Video/docs/V_0/README.md`。此名称不改变Qwen模型版本。

# 当前发音与停顿要求

先读 [SPEECH_POLICY.md](docs/SPEECH_POLICY.md)，2026-10-27最新词组读音、模型名内部短停顿和句间停顿以此为准；旧样片不改。

# 本地新闻语音生成：Qwen3-TTS 0.6B

当前主后端为 **Qwen3-TTS-12Hz-0.6B-Base**。用户已试听确认更喜欢 Qwen 的效果。当前通过参考音频克隆音色，**尚未执行 Qwen 微调训练**，不能把零样本克隆写成“训练完成”。

本项目提供本地调用、新闻逐句时间映射、参考音频复用、数据准备和试听对照。模型与训练框架来自 [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)。Qwen 的官方代码/权重许可按其 Apache-2.0 文件执行；vendor/qwen3_official 保留上游版权和 LICENSE。历史 GPT-SoVITS 部分来自 RVC-Boss/GPT-SoVITS，沿用其原有 MIT 许可与来源说明。

## 当前选择与硬件

- RTX 4060 Laptop，8GB 总显存；本次检查约 4.4GB 空闲。
- 用户确认：若 1.7B 显存不足就放弃，0.6B 效果已好。因此不在本机强行启动 1.7B 常规全参数训练，也不为此中断其他 GPU 工作。
- 本次下载产生的 1.7B 目录已清理；没有训练出 1.7B 结果。0.6B 官方基础权重、参考音频和现有可用环境保留。
- 延长轮数不会降低显存需求，也不保证自然度提升。未来确需优化 0.6B 时，先补充自然叙述素材、试跑显存、以留出验证和试听选模型，不机械选择最后一轮。

## 调用

新闻项目实际路径为 `../Codex-report-news/News_Video`，默认 `config.voice.backend=qwen3`。

```powershell
& ./.venv-qwen3/Scripts/python.exe -B ./tools/qwen_generate_context.py --job JOB_JSON
```

job 必需字段：model_path、reference、reference_text、text_file、output_file。使用本地 0.6B Base 目录和逐字准确的参考台词；新闻流程已接入逐句生成、2 秒导览停顿及字幕时间映射。默认保留自然尾字，不压缩/淡出或机械变速。不要把旧 GPT-SoVITS 的采样/速度参数当成 Qwen 已实现的参数。

环境使用专用 `.venv-qwen3`，只读复用本机 CUDA PyTorch；不将这个目录直接当作跨机可运行包。固定 Qwen 依赖见 requirements-qwen3.txt。新机器另行安装运行时、下载官方权重和准备参考音频，不上传个人音频/权重/环境到 Git。

当前样音与同文稿对照在本机 `experiments/qwen3-20261027/`。详细接入说明见 [QWEN3_TTS_INTEGRATION.md](docs/QWEN3_TTS_INTEGRATION.md)。历史桌面入口“启动语音生成.bat”仍为 GPT-SoVITS 工具，不是 Qwen 界面；新闻工程和上面的 Qwen 入口才使用当前默认后端。

## 数据与后续训练

现有素材共 32 条，已导出 28 条训练、4 条留出验证；每条必须有准确的 audio/text/ref_audio。原始素材、标注、参考音频和数据清单保留。已有格式导出不代表完成采样率归一化、音频码提取或模型训练，正式训练前还需将音频和参考规范为 24kHz 并验证。

官方单说话人微调流程及本地准备工具：

- `tools/prepare_qwen_dataset.py`：导出清单、检查重复和训练/验证泄漏。
- `vendor/qwen3_official/`：带原始许可的官方数据与准备脚本，仅作为已核查的上游参考；训练改进须在独立目录进行。
- 当前不启动长时间训练；保留 0.6B 可用音色。未来训练按阶段提高上限，保留最佳验证模型和最后断点，出现漏字、加速、音色退化即停止评估。

## 旧训练产物的整理

17 个不被当前配置引用的 GPT-SoVITS 检查点已移至工作区 `.retired-models/20261027-gpt-sovits-checkpoints/`，约 5.30GiB。manifest.json 保存原路径、归档路径、字节数和 SHA-256，移动后已校验；这是可恢复归档，**没有声称永久删除或释放这部分磁盘空间**。需要恢复旧实验时按清单还原，不能直接运行引用已归档路径的历史实验 profile。

原始音频、参考、标注、当前 0.6B、代码与实验记录不清理。GPT-SoVITS 原安装位于工作区外，仍保留，以免擅自影响其他项目；已发布成品保持不变。历史说明见 [legacy-gpt-sovits-readme.md](docs/legacy-gpt-sovits-readme.md)。

## 验证边界

已验证：Qwen 本地 CUDA 克隆、新闻逐句与导览停顿、字幕时间映射，用户偏好 0.6B；未验证或未做：1.7B 训练、Qwen 微调效果、更多轮数一定改善自然度。


时间口播规范：13点48分 / 13:48 读“十三点四十八分”，48 不逐位读“四八”。仅转换语音稿，画面/字幕保留原有阿拉伯数字；分钟小于10时读“零五”等，整点读“点整”。tibo 目标为梯一声、啵轻声、两音节连读；语气平稳克制，整段统一生成。


用户最新语速要求：在当前基准上仅将说话部分加快约3%（speech_rate=1.03），每句处理后再插入原长停顿，pause仍为0.20秒，不将整段含停顿音频一起加速。FFmpeg atempo保持音高，禁止额外吞尾或压缩句尾。导览2秒静音另按原规则保留。发音目标为梯一声/啵轻声；时间为十三点四十八分，字幕仍用数字。采样设置是稳定性调整，不是已验证的情绪控制，最终听感待试听。


## 当前生效：连贯语音与停顿（2026-10-27 最新）

- 主模型仍为用户认可的 Qwen3-TTS 0.6B Base，未重新训练或替换权重。
- 改为同一条新闻/场景连续生成，再用本地 FunASR 文本对齐和波形低能量句界恢复逐句时间。避免每句话单独重新采样导致的语气变化；字幕仍单行，tibo 与数字的显示稿不改。
- 取消额外 3% 加速，speech_rate=1.0。句间目标静音约 0.40 秒，场景/新闻切换约 0.60 秒；导览后的 2 秒留时仍按场景规则执行。只调整检测到的低能量间隙，不对发音尾字压缩或淡出；找不到可靠间隙就报错，不盲切活动语音。
- 保留发音目标：tibo 为梯一声、啵轻声（语音稿梯啵），13点48分为十三点四十八分，画面和字幕仍用 tibo/数字。
- 正式新闻入口根据 config.voice.qwen3.contextual_generation 调用 Voice_Generate/tools/qwen_generate_context.py；底层单句接口仍为 qwen_generate_job.py。对齐模型在 Voice_Generate/models/fa-zh，Python 在 config 的 alignment_python；只复用旧环境里的本地 ASR 依赖，不读取旧 GPT-SoVITS 音色 profile/权重。新机器需准备运行时及官方对齐模型，模型不进 Git/ZIP。
- 技术验证：18 项语音测试、54 项新闻测试通过；实际正式入口的 3 句隔离烟测通过，包含 2 秒导览留时、逐句时间映射且移除旧 GPT profile/engine 字段仍可运行。新样片为两段新闻上下文、8 句对齐输出；辅助 ASR 为十三点四十八分，识别不能替代轻声/自然度试听。
- 若新流程仍达不到用户听感要求，再训练独立的 0.6B 候选，使用留出验证与同文稿试听选模型，通过后切换 model_path；不直接覆盖可用 Base，不单纯堆轮数。当前尚未执行 Qwen 微调。

详细上下文接口及本地对齐说明见 [CONTEXTUAL_VOICE.md](docs/CONTEXTUAL_VOICE.md)。context job 除原字段外还需要 alignment_python、alignment_model；可用 context_group_sentence_counts 明确逐新闻分组。底层 qwen_generate_job.py 保留用于独立单句测试。

2026-10-29词组读音：按用户明确要求，干员读gàn yuán（干四声）；tools/speech_text.py仅TTS输入将干员替换为同音赣员，字幕与原文仍为干员，其他干字词不变。新增干员/六星干员/干员展示适用，不改模型、音色、语速和停顿。实际声调须试听，ASR不能单独证明。
