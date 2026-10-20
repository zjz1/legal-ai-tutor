# ⚖️ 智能法律学习与咨询助手 (Legal AI Tutor)

本项目是一个面向法学学习与实务咨询的智能辅助工具。结合了**本地部署的微调大模型**与**云端大模型 API**，设计了“本地/专家双模式”以及针对法律实务的“学习与分析双功能”。

## 🌟 核心特性

- **本地模式 (Local Mode)**：基于 Ollama 部署微调后的 `Legal-Qwen-7B` 模型，保障数据隐私与离线可用性。
- **专家模式 (Expert Mode)**：构建“本地预处理 → 云端生成 → 本地风格调优”的链路，在保障隐私的前提下利用云端大模型提升复杂法律问题的回答质量。
- **学习模式 (Learning Mode)**：在回答中自动拆解法律概念、串联知识点，生成适合法学学习的知识网络。
- **实际情况分析 (Case Analysis)**：实现“现实问题 → 法律要素提取 → 法律解答”的两层处理，模拟律师接案思维。

## 🏗️ 项目架构

- `data/`: 包含微调数据集清洗脚本（提取 DISC-Law-SFT 数据）。
- `src/workflow.py`: 基于 Python 实现的核心双模式路由与 LangGraph 逻辑。
- `src/local_api.py`: 通过 HTTP 请求调用本地 Ollama API 的封装。
- `models/`: 存放模型的目录（需自行下载，见下文）。

## 🚀 快速开始

### 1. 下载微调模型
本项目配套的 QLoRA 微调模型（GGUF 格式，约 4.7GB）已发布至 ModelScope：
👉 [ModelScope 模型下载地址](https://www.modelscope.cn/models/nanzjz1/legal-qwen-7b-gguf)

使用 Ollama 一键拉取并运行：
    ollama run modelscope.cn/nanzjz1/legal-qwen-7b-gguf

### 2. 运行代码环境
请确保已安装 Python 3.10+，并启动 Ollama 服务。

    pip install -r requirements.txt
    python src/workflow.py

## 📊 微调效果对比 (A/B 测试)

为了验证微调有效性，我们构建了原版 Qwen2.5-7B 与微调模型的 A/B 测试：
1. **针对性的格式对齐**：原版模型易混淆“诉请”与“查明事实”且输出冗长；微调版能够更精准地输出 【争议焦点】、【法院查明】、【判决结果】 等标准结构，有助于降低后续文本清洗的成本。
2. **一定的法言法语转换能力**：微调版在案情口语转书面语任务中，能生成包含【诉讼请求】等规范要素的文本，并能在特定场景下引用《民法典》相关条款。

## ⚖️ 许可与致谢

本项目基于以下开源项目构建，特此致谢：
- **基座模型**：Qwen2.5-7B-Instruct (Apache 2.0) - 阿里巴巴通义千问团队。
- **微调数据集**：DISC-Law-SFT (Apache 2.0) - 复旦大学数据智能与社会计算实验室 (Fudan-DISC)。

如果本研究的模型或方法对您的研究和工作有帮助，请考虑引用原作者的论文：

    @misc{yue2023disclawllm,
      title={DISC-LawLLM: Fine-tuning Large Language Models for Intelligent Legal Services},
      author={Shengbin Yue and Wei Chen and Siyuan Wang and Bingxuan Li and Chenchen Shou and Shujun Liu and Yuxuan Zhou and Yao Xiao and Song Yun and Xuanjing Huang and Zhongyu Wei},
      year={2023},
      eprint={2309.11325},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
    }
