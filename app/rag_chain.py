# app/rag_chain.py
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml
from langchain_core.prompts import ChatPromptTemplate

from .llm_client import LLMFactory
from .vectorstore import load_vectorstore
from .config import config


def load_prompts() -> Dict[str, Any]:
    prompt_path = Path(config.prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt 文件不存在: {prompt_path}")

    with prompt_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class DebugRAGChain:
    """
    - similarity_search_with_score：显式用 query 做 embedding + 相似度
    - 打印 retrieved 文档和分数
    """

    def __init__(self, vectorstore, prompt: ChatPromptTemplate, llm):
        self.vectorstore = vectorstore
        self.prompt = prompt
        self.llm = llm

    def invoke(self, question: str) -> str:
        docs_with_scores: List[Tuple] = self.vectorstore.similarity_search_with_score(
            question,
            k=4,
        )

        print("\n====== [RAG] 检索阶段：Retrieved Documents (with scores) ======")
        if not docs_with_scores:
            print("未检索到任何文档。")
        else:
            for i, (doc, score) in enumerate(docs_with_scores, start=1):
                source = doc.metadata.get("source", "")
                print(f"\n[doc{i}] score={score:.4f} source={source}")
                print(doc.page_content)
                print("-" * 60)

        # 把 docs 抽出来用于拼接上下文
        docs = [d for d, _ in docs_with_scores]

        context = "\n\n".join(
            f"[doc{i}] {doc.page_content}" for i, doc in enumerate(docs, start=1)
        )

        # 构造 Prompt
        messages = self.prompt.format_messages(context=context, question=question)

        print("\n====== [RAG] 构造 Prompt：发送给 LLM 的完整消息 ======")
        for msg in messages:
            role = getattr(msg, "role", getattr(msg, "type", ""))

        print("\n====== [RAG] 等待模型回复中... ======")
        resp = self.llm.invoke(messages)
        content = getattr(resp, "content", str(resp))

        return content


def create_rag_chain() -> DebugRAGChain:
    prompts = load_prompts()
    rag_prompt_cfg = prompts["rag_qa"]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", rag_prompt_cfg["system"]),
            ("user", rag_prompt_cfg["user"]),
        ]
    )

    vectorstore = load_vectorstore()
    llm = LLMFactory().create_chat_model()

    return DebugRAGChain(vectorstore, prompt, llm)