# 管理员模型重新训练与对照

日期：2026-10-26。

## 已完成

SoVITS 完成 12 轮，保存 e4/e8/e12；GPT 整理音频语义版本完成 10 轮，保存 e2/e4/e6/e8/e10；GPT 保留原始节奏版本完成 6 轮，保存 e2/e4/e6。所有训练从预训练底模开始，并非把旧管理员权重改名。原始录音 SHA-256 复核保持不变。

32 条素材按固定索引分成 28 条训练和 4 条留出。SoVITS 使用适度缩短长停顿后的波形，并重提 HuBERT/SV/语义特征。BERT 只依赖原文本，复用一致的原特征。节奏试验的 GPT 使用原始语义序列，但仍严格过滤到同样的 28 条训练记录。旧模型原来用过全部 32 条，因此留出源台词对旧模型不是独立留出，比较结论不应解释为严格泛化排名。

## 修复与验证

- 单卡不初始化 DDP/TCPStore；未初始化进程组时采样器采用单设备参数。
- 训练入口显式优先加载隔离副本，避免整合包 Python 忽略 PYTHONPATH 后回退到旧代码。
- GPT 学习率不再锁在 0.002。实际训练日志显示 warmup/cosine 变化，与配置相符。
- 中文 checkpoint 通过 Python 文件对象写入，同目录原子替换；配置显式 UTF-8。
- num_workers=0 时关闭 persistent workers 和预取。
- 仓库 4 项回归测试通过；可移植 prepare 命令（含保留原节奏分支）独立跑通；train --stage s1 完成 1 轮并保存 checkpoint；synthesize 已用部署后的 profile 生成新音频。

## 比较方法和结果

第一轮：旧模型与 9 组新检查点组合，3 种未见测试文本、同一随机种子，共 30 段。第二轮：下表 6 组模型、同样 3 种文本、两个新种子 7/20261027，共 36 段。共生成并识别 66 个新对照样本。两轮均固定 ref_00 参考音频、temperature=0.75、top_k=5、top_p=1、cut5、语速 1、非流式。

| 模型组合 | 样本数 | ASR CER | 声纹余弦均值 | 中位 RTF |
|---|---:|---:|---:|---:|
| old_g15_s8 | 6 | 0.43% | 0.881 | 0.473 |
| new_g2_s12 | 6 | 0.00% | 0.852 | 0.477 |
| timing_g2_s12 | 6 | 0.00% | 0.856 | 0.452 |
| timing_g4_s12 | 6 | 0.00% | 0.855 | 0.558 |
| timing_g6_s12 | 6 | 0.00% | 0.857 | 0.538 |
| hybrid_oldg15_news12 | 6 | 0.00% | 0.879 | 0.526 |


命名：old_g15_s8 是旧 GPT e15 + 旧 SoVITS e8；new_g2_s12 是整理后数据训练的 GPT e2 + 新 SoVITS e12；timing_gN_s12 是保留原节奏的 GPT eN + 新 SoVITS e12；hybrid_oldg15_news12 是旧 GPT e15 + 新 SoVITS e12。

## 选择

推荐候选为 hybrid_oldg15_news12：保留更贴近已有角色语气的旧 GPT，搭配新训练的 SoVITS e12。该轮 6 个样本 ASR 没有识别差异，声纹指标与旧模型很接近。另提供完整新训练的 timing_g6_s12，满足两阶段都重新训练的使用需求。保留 baseline 入口，不覆盖旧权重。

不能宣称新模型全面胜过旧模型：完整新训练版本的声纹指标仍稍低；旧模型的非零 CER 只涉及少量识别差异，不足以证明稳定性显著不同。ASR 自带错误，SV 使用项目自身的 ERes2NetV2，也不是独立盲测。自然度、电音和情绪表现尚未人工试听确认。耗时受冷启动和运行波动影响，没有证据声称稳定提速。

## 本机交付记录

- recommended_profile.json：旧 GPT + 新 SoVITS，推荐候选。
- full_new_profile.json：保留节奏的新 GPT e6 + 新 SoVITS e12。
- baseline_profile.json：原来的 GPT e15 + SoVITS e8。
- comparison/comparison.html：30 段检查点筛选。
- validation/comparison.html：36 段新种子验证，含原文和 ASR 结果。
- GPT_weights/、SoVITS_weights/、timing_trial/GPT_weights/：训练完成的权重。
- deployment_backup/：原界面与配置备份；默认选择改动可以恢复。

原 WebUI 默认模型及 API 配置已设为 recommended，完整新模型已放入原 weights 文件夹供选择。生成入口只要更换 profile 即可切换。没有启动或操作正在打开的 WebUI 窗口，默认选项需要下次启动时生效。

## 代码来源

优化代码上传至 https://github.com/OTAXIO/Voice_Generate 。参考上游为 https://github.com/RVC-Boss/GPT-SoVITS ，本地依据为 GPT-SoVITS-v2pro-20250604 整合包；精确上游 commit 未验证，不作虚构对应。上游 MIT License 和原文件注释保留。录音、权重和运行缓存不上传。
