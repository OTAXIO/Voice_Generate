# Voice_Generate

围绕 **GPT-SoVITS v2Pro** 的 Windows 单卡训练修复、数据准备与新旧模型对照工具。主要用于已有标注和特征工程的角色声音微调。

## 上游来源与致谢

本项目参考并修改自 **[RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)**，原作者和贡献者完成了语音模型、训练框架和推理引擎的主体工作。本仓库提供兼容性修复和工作流工具，不是独立开发的 TTS 引擎，也不是 GPT-SoVITS 官方发布。

本次修改依据为本地 `GPT-SoVITS-v2pro-20250604` 整合包。该目录包含定制代码，**未核实其对应的精确上游 commit**，因此不将目录日期当作已验证的官方版本号。[upstream_manifest.json](upstream_manifest.json) 记录修改前后文件的 SHA-256，便于核对来源。

`overrides/GPT_SoVITS/` 中的文件为上游衍生代码，保留原文件中的来源注释；其中 AR 模块原有的 SoundStorm、VALL-E 等引用也予以保留。沿用上游 **MIT License / Copyright (c) 2024 RVC-Boss**，完整许可见 [LICENSE](LICENSE)。模型权重和数据另依各自来源许可取得。

## 本次修复

- **单卡通信**：SoVITS 单设备直接训练，多 GPU 才启动分布式通信，避免单卡仍依赖 TCPStore。GPT 单卡使用普通训练策略，采样器在未初始化进程组时采用 world size 1、rank 0。
- **模块加载**：训练入口显式优先加载运行目录中的修复副本，兼容整合包隔离 Python 忽略 `PYTHONPATH` 的情况。
- **学习率**：修复本地版本把 GPT 学习率锁死为 `0.002`、调度器不采用计算值的问题；初始化、warmup、cosine decay 和报告值一致。
- **文件保存**：checkpoint 临时文件放在目标目录，用文件对象保存，再原子替换；避免在安装目录写临时文件及中文路径问题。
- **编码和 DataLoader**：配置显式按 UTF-8 读写；`num_workers=0` 时关闭 persistent workers，并不设置预取因子。
- **可追溯实验**：独立输出目录、训练/留出划分、重提音频特征、多个检查点、同条件 ASR / 声纹 / 耗时比较。

## 目录

```text
voice_generate.py                 # 使用整合包解释器运行各命令
tools/prepare.py                 # 整理音频、生成新特征与训练配置
tools/train.py                   # 隔离训练，支持阶段选择和 STOP 标记
tools/compare.py                 # 同参考、同文本、同参数的新旧对照
tools/synthesize.py              # 从显式 profile 生成音频
tools/set_default.py             # 备份后统一 WebUI、缓存和 API 的默认声音
overrides/GPT_SoVITS/            # 优化后的上游衍生源码
examples/                       # 配置格式示例，不含声音或权重
docs/                           # 训练参数、验证结果与限制
tests/test_runtime_fixes.py       # 回归测试
```

录音、模型、运行日志、缓存和安装包不在本仓库内。工具在 `--output` 指定的目录工作，不自动删除素材或改写原安装目录。

## 环境和使用前提

已在 Windows、Python 3.9.13、PyTorch 2.0.0+cu118、RTX 4060 Laptop 8 GB 上完成训练验证。沿用本地 GPT-SoVITS 整合包的解释器和依赖；本仓库不附带运行时、预训练模型或一键重新安装环境脚本。

需要已经具备：

1. 可用的 v2Pro 预训练 GPT / SoVITS / HuBERT / 中文 BERT / SV 模型；比较工具还需要本地 Paraformer Large。
2. `audio_path|speaker|language|text` 格式的 UTF-8 `.list`。
3. 与上述文本一致的既有工程 `2-name2text.txt` 和 `3-bert/`。BERT 特征只依赖文本，本次可复用；音频变更后重新生成 HuBERT、SV 和语义特征。

此版本按上述本地分支验证。其他 GPT-SoVITS 版本或多卡环境尚未验证，不能把覆盖文件当作所有上游版本通用的补丁。

## 准备与训练

在仓库目录运行，路径请替换成自己的安装位置：

```powershell
$project = 'D:\Tools\GPT-SoVITS-v2pro-20250604'
$python = "$project\runtime\python.exe"

& $python -B .\voice_generate.py prepare --project $project --output .\runs\admin `
  --labels "$project\output\gly_asr_opt\gly_slicer_opt.list" `
  --features "$project\logs\管理员" --holdout 3,10,19,23

& $python -B .\voice_generate.py train --project $project --output .\runs\admin
```

`--holdout` 为 `.list` 中从 0 开始的记录索引，默认值对应本次 32 段素材。换素材时应重新决定留出记录。准备过程会转换为 32 kHz 单声道，保留约 120 ms 外侧余量，把超过 0.8 秒的内部低能量停顿缩至约 0.45 秒；判定阈值为 20 ms 帧的 −40 dBFS RMS。这是实验处理规则，不能把所有低能量部分都视为无用静音。

默认训练参数：SoVITS 12 轮、batch 4、LR `5e-5`，每 4 轮保存；GPT 10 轮、batch 4，LR 从 `1e-5` warmup 到 `1e-4`，cosine 衰减到 `1e-5`，每 2 轮保存。配置保存在输出目录的 `s2.json`、`s1.yaml`，训练前可以修改。上游 GPT 手动优化会按多个 batch 累积梯度，batch size 不等于一次 optimizer update 的独立样本数。

准备时可加 `--gpt-original-timing`，让 SoVITS 使用整理后的音频，而 GPT 使用原有语义特征保留台词节奏；此时 `--features` 还须包含原始 `6-name2semantic.tsv`。两边保持相同训练/留出索引。这是本次做过的额外消融实验，不能预先保证比统一整理数据更好。

只重跑一个阶段可以使用 `--stage s1` 或 `--stage s2`。训练器会尝试读取该运行目录的已有 checkpoint；需要完全重新训练时请使用新输出目录。创建 `runs/admin/STOP` 可停止当前训练子进程树；恢复前应显式移除 STOP。停止不保证保存当前未完成 batch，但保留已完成的检查点。

## 比较和生成

编辑 [examples/comparison.json](examples/comparison.json)，填入新旧权重、参考音频和测试文本的**绝对路径**。两组模型使用相同参考、参数和种子，避免模型、参考音频一起变化导致无法判断原因。

```powershell
& $python -B .\voice_generate.py compare --project $project --output .\runs\comparison `
  --spec .\examples\comparison.json
```

输出 `summary.json`、完整 `comparison.json`、WAV 以及可离线打开的 `comparison.html`。生成和 ASR 分阶段执行以控制显存。比较参数默认为中文 `all_zh`、temperature 0.75、top_k 5、top_p 1、cut5、语速 1、非流式；英文和混合语种需要另行配置和验证。

编辑 [examples/profile.json](examples/profile.json) 后可明确指定声音生成：

```powershell
& $python -B .\voice_generate.py synthesize --project $project --output .\runs\sample `
  --profile .\examples\profile.json --text '今天的工作已经完成，我们明天再继续。'
```

若希望原 WebUI 也默认选中该声音，先把两份权重放到安装目录对应的 weights 文件夹，再执行：

```powershell
& $python -B .\voice_generate.py set_default --project $project --output .\runs\webui-backup `
  --profile .\examples\profile.json
```

这一步会明确修改原安装目录的 `webui.py` 默认选项、`weight.json` 和 `tts_infer.yaml`，原文件备份在 `--output`。其他命令只在输出目录工作。如此可避免“配置文件选了管理员，主界面下拉框却又覆盖成其他模型”的问题。

## 测试与结果

```powershell
& $python -B .\tests\test_runtime_fixes.py -v
```

4 项回归测试覆盖学习率实际变化、中文 checkpoint 往返读写、CUDA 可用但未初始化分布式组的采样器，以及 WebUI/API/缓存默认模型一致性。训练结果见 [docs/experiment.md](docs/experiment.md)。

CER 包含 ASR 自身错误；声纹余弦使用项目自带 SV 编码器，并不是独立盲测，也不是“像本人百分比”。这些数值不能证明自然度、情绪或电音已经改善，必须结合试听。训练完成、自动指标通过与人耳更喜欢某个版本是不同的结论；本项目会分别记录。


## Qwen3-TTS 候选后端

用户已选 Qwen3-TTS，并接入新闻默认后端；原 GPT-SoVITS 仍保留回退。接口为 tools/qwen_generate_job.py --job JOB_JSON，使用 .venv-qwen3/Scripts/python.exe；job 格式示例在本机 experiments/qwen3-20261027/sample-job.json，不能将音频/权重/环境目录上传 Git。安装和训练说明见 docs/QWEN3_TTS_INTEGRATION.md。真实推理与新闻时间映射已通过，音色自然度待试听，不代表已完成新模型训练。
