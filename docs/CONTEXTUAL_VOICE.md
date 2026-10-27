## 当前生效：连贯语音与停顿（2026-10-27 最新）

- 主模型仍为用户认可的 Qwen3-TTS 0.6B Base，未重新训练或替换权重。
- 改为同一条新闻/场景连续生成，再用本地 FunASR 文本对齐和波形低能量句界恢复逐句时间。避免每句话单独重新采样导致的语气变化；字幕仍单行，tibo 与数字的显示稿不改。
- 取消额外 3% 加速，speech_rate=1.0。句间目标静音约 0.40 秒，场景/新闻切换约 0.60 秒；导览后的 2 秒留时仍按场景规则执行。只调整检测到的低能量间隙，不对发音尾字压缩或淡出；找不到可靠间隙就报错，不盲切活动语音。
- 保留发音目标：tibo 为梯一声、啵轻声（语音稿梯啵），13点48分为十三点四十八分，画面和字幕仍用 tibo/数字。
- 正式新闻入口根据 config.voice.qwen3.contextual_generation 调用 Voice_Generate/tools/qwen_generate_context.py；底层单句接口仍为 qwen_generate_job.py。对齐模型在 Voice_Generate/models/fa-zh，Python 在 config 的 alignment_python；只复用旧环境里的本地 ASR 依赖，不读取旧 GPT-SoVITS 音色 profile/权重。新机器需准备运行时及官方对齐模型，模型不进 Git/ZIP。
- 技术验证：18 项语音测试、54 项新闻测试通过；实际正式入口的 3 句隔离烟测通过，包含 2 秒导览留时、逐句时间映射且移除旧 GPT profile/engine 字段仍可运行。新样片为两段新闻上下文、8 句对齐输出；辅助 ASR 为十三点四十八分，识别不能替代轻声/自然度试听。
- 若新流程仍达不到用户听感要求，再训练独立的 0.6B 候选，使用留出验证与同文稿试听选模型，通过后切换 model_path；不直接覆盖可用 Base，不单纯堆轮数。当前尚未执行 Qwen 微调。

实现依据：FunASR 官方时间戳预测接口 https://github.com/modelscope/FunASR/blob/main/examples/industrial_data_pretraining/paraformer/README_zh.md 。本地静音边界修正与样片验证为本项目实现，不声称逐词时间绝对精确。


最新发音/复合名停顿规则见 [SPEECH_POLICY.md](SPEECH_POLICY.md)，单词内部约.12秒与句间.4秒独立。
