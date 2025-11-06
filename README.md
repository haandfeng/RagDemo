# RagDemo —— 基于 LangChain + Chroma + SiliconFlow 的本地 RAG Demo

## 一、项目简介

这个项目是一个最小可用的 RAG Demo，特点：

- 使用 **SiliconFlow** 的 OpenAI 兼容接口
  - 对话模型：`Qwen/Qwen3-8B`
  - Embedding 模型：`BAAI/bge-m3`
- 向量库：本地 **Chroma**，数据保存在 `chroma_data/`
- Prompt：使用 **YAML** 管理，放在 `prompts/rag_prompts.yaml`
- 使用 **LangChain** 进行 LLM/Embedding 抽象和检索
- 内置 Debug 输出：
  - 打印检索出来的文档 + 相似度分数
  - 打印最终喂给 LLM 的完整 Prompt（system/user）
  - 打印“等待模型回复”和模型原始回复，方便调试

## 二、目录结构

```text
RagDemo/
  ├─ app/
  │   ├─ __init__.py
  │   ├─ config.py            # 读取 config.yaml，统一管理路径和模型配置
  │   ├─ embeddings.py        # 封装 BAAI/bge-m3 embedding（通过 SiliconFlow）
  │   ├─ llm_client.py        # 封装 Qwen/Qwen3-8B 对话模型（通过 SiliconFlow）
  │   ├─ vectorstore.py       # Chroma 向量库构建和加载
  │   ├─ rag_chain.py         # 带 Debug 输出的 RAG 调用链
  │   └─ main.py              # 程序入口：构建向量库 + 交互问答
  ├─ prompts/
  │   └─ rag_prompts.yaml     # RAG 用的 Prompt 模板（YAML）
  ├─ chroma_data/             # Chroma 本地数据目录（运行后自动生成）
  ├─ config.yaml              # 项目配置（API Key、模型名、路径等）
  └─ requirements.txt
```
## 三、运行方式

请首先在config.yaml里配置自己的硅基流动api:  api_key: "sk-xxxx"
```bash
cd /path/to/RagDemo
python app.main
```
程序会做两件事：
1.	构建向量库（只在第一次或 chroma_data 为空时运行）
- 使用 app/main.py 中的 ingest_demo_docs() 将示例文本写入 Chroma
2. 启动交互命令行： 
- 输入问题和系统对话
- exit 或 quit 退出
