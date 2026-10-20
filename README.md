---
license: apache-2.0
language:
  - zh
tags:
  - legal
  - qwen2.5
  - gguf
  - ollama
  - text-generation
base_model: Qwen/Qwen2.5-7B-Instruct
datasets:
  - AI-ModelScope/DISC-Law-SFT
---

# ⚖️ Legal Qwen 7B (法律 AI 助手微调版)

基于 Qwen2.5-7B-Instruct 基座模型，使用 500 条真实裁判文书数据，通过 QLoRA (4-bit) 微调并导出为 GGUF (Q4_K_M) 格式的法律垂直领域大模型。模型完美适配 Ollama 本地部署，兼顾专业度与数据隐私。

## 🌟 核心亮点 (A/B 测试对比)

相比原版 Qwen2.5-7B，微调后的模型在垂直法律场景下展现出了显著的“工作流契合度”：
1. **极强的格式对齐能力**：在“判决书结构化摘要”任务中，原版模型容易混淆“诉请”与“查明事实”且输出冗长；微调版能够精准、极简地输出 【争议焦点】、【法院查明】、【判决结果】 等标准结构，极大降低法律文本清洗成本。
2. **专业的法言法语转换**：在“案情口语转书面语”任务中，原版模型存在角色错位和语病；微调版可直接生成符合规范的【诉讼请求】，并能主动引用《民法典》条款（如判令被告支付欠款及利息），具备极强的零样本法条召回能力。
3. **本地隐私与轻量化部署**：经过 Q4_K_M 量化，仅需约 4.7GB 显存即可在本地离线流畅运行，彻底杜绝法律敏感数据的云端泄露风险。

---

## 🚀 快速开始 (Quick Start)

推荐使用 **Ollama** 进行本地一键拉取和运行。

**方法一：直接通过 ModelScope 源运行**

    ollama run modelscope.cn/nanzjz1/legal-qwen-7b-gguf

**方法二：手动配置 Modelfile**

1. 下载本仓库中的 `legal-qwen-q4.gguf` 文件。
2. 在同级目录下创建名为 `Modelfile` 的文件（无后缀），填入以下内容：

    FROM ./legal-qwen-q4.gguf

    TEMPLATE """{{ if .System }}<|im_start|>system
    {{ .System }}<|im_end|>
    {{ end }}{{ if .Prompt }}<|im_start|>user
    {{ .Prompt }}<|im_end|>
    {{ end }}<|im_start|>assistant
    """

3. 执行创建与运行命令：

    ollama create legal-qwen -f Modelfile
    ollama run legal-qwen

---

## 🤝 开源项目地址
本项目配套的 LangGraph 双 Agent 工作流（支持本地模式+专家模式）及 500 条微调数据集已同步开源至 GitHub：
🔗 [https://github.com/zjz1/legal-ai-tutor](https://github.com/zjz1/legal-ai-tutor)

---

## ⚖️ 许可与致谢 (License & Acknowledgements)

本模型基于以下优秀的开源项目构建，特此致谢并严格遵守其开源许可协议：

### 1. 基座模型 (Base Model)
- **Qwen2.5-7B-Instruct**，由阿里巴巴通义千问团队开源。
- **许可证**：Apache License 2.0。
- **版权声明**：Copyright (c) Alibaba Cloud. All Rights Reserved.

### 2. 微调数据集 (Fine-tuning Dataset)
- **DISC-Law-SFT**，由复旦大学数据智能与社会计算实验室 (Fudan-DISC) 开源。
- **许可证**：Apache License 2.0。
- **引用**：如果本研究的模型或方法对您的研究和工作有帮助，请考虑引用原作者的论文：

    @misc{yue2023disclawllm,
      title={DISC-LawLLM: Fine-tuning Large Language Models for Intelligent Legal Services},
      author={Shengbin Yue and Wei Chen and Siyuan Wang and Bingxuan Li and Chenchen Shen and Shujun Liu and Yuxuan Zhou and Yao Xiao and Song Yun and Xuanjing Huang and Zhongyu Wei},
      year={2023},
      eprint={2309.11325},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
    }