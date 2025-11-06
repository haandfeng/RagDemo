# app/main.py
from pathlib import Path

from app.vectorstore import build_vectorstore_from_texts
from app.rag_chain import create_rag_chain
from app.config import config


# app/main.py 片段
from pathlib import Path

from app.vectorstore import build_vectorstore_from_texts
from app.rag_chain import create_rag_chain
from app.config import config


def ingest_demo_docs():
    """
    如果 chroma_data 已存在且非空，就跳过重新 embedding。
    """
    chroma_dir = Path(config.vectorstore.persist_dir)

    if chroma_dir.exists() and any(chroma_dir.iterdir()):
        print(f"✅ 检测到已存在向量库目录：{chroma_dir}，跳过重新 embedding。")
        return

    chroma_dir.mkdir(parents=True, exist_ok=True)

    # 🔹 这里放 6 段风格差异较大的文档，方便测试检索
    texts = [
        # 1. 硅基流动
        "硅基流动（SiliconFlow）是一家提供大模型推理服务的公司，"
        "其平台兼容 OpenAI 接口，用户可以用相同的 SDK 和协议调用包括 Qwen 在内的多种中文模型。",

        # 2. Qwen3-8B 对话模型
        "Qwen3-8B 是通义千问系列中的一款中等规模开源模型，适合做对话、代码编写和基础问答等任务，"
        "在中文场景下有较好的效果，常搭配 RAG 方案使用。",

        # 3. BAAI/bge-m3 向量模型
        "BAAI/bge-m3 是智谱和北京智源发布的多语言通用向量模型，"
        "支持中文、英文等多种语言，常用于语义检索、RAG、重排序等场景。",

        # 4. RAG 介绍
        "RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合向量检索与大模型生成的技术方案，"
        "它通过先从向量数据库中检索相关文档，再把文档和问题一起喂给大模型，从而减少幻觉、增强对知识库的利用。",

        # 5. Python + LangChain 开发
        "在 Python 中可以使用 LangChain 快速搭建 RAG 应用，"
        "比如使用 Chroma 作为本地向量库、OpenAIEmbeddings 作为 embedding 封装、ChatOpenAI 作为对话模型封装。",

        # 6. 完全不相关的旅游段落
        "日本京都是一座历史悠久的城市，拥有清水寺、金阁寺等世界文化遗产。"
        "春天可以赏樱，秋天可以观红叶，是非常热门的旅游目的地。",
    ]
    metadatas = [
        {"source": "siliconflow_intro"},
        {"source": "qwen3_8b_intro"},
        {"source": "bge_m3_intro"},
        {"source": "rag_intro"},
        {"source": "python_langchain_intro"},
        {"source": "kyoto_travel"},
    ]

    build_vectorstore_from_texts(texts, metadatas)
    print(f"✅ 已完成示例文档向量化并写入 Chroma：{chroma_dir}\n")


def interactive_chat():
    rag_chain = create_rag_chain()

    print("RAG Debug Demo 已启动，输入问题开始对话，输入 `exit` 退出。")
    while True:
        question = input("\n用户: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        try:
            # DebugRAGChain.invoke 内部会打印检索 / Prompt / 等待 / 回复
            answer = rag_chain.invoke(question)

            print("\n====== 最终回答（整理后给用户） ======")
            print(answer)
            print("====================================\n")
        except Exception as e:
            print(f"发生错误: {e}")


if __name__ == "__main__":
    ingest_demo_docs()
    interactive_chat()