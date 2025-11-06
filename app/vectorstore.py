# app/vectorstore.py
from typing import Iterable, List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import config
from .embeddings import create_embedding_model


def build_vectorstore_from_texts(
    texts: Iterable[str],
    metadatas: Iterable[dict] | None = None,
) -> Chroma:
    """
    将原始文本切分后写入本地 Chroma。
    Chroma 的 persist_directory 已统一配置为 RagDemo/chroma_data。
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"],
    )

    docs: List[Document] = []
    for i, text in enumerate(texts):
        meta = {} if metadatas is None else (metadatas[i] if i < len(metadatas) else {})
        for chunk in text_splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata=meta))

    embeddings = create_embedding_model()
    vs_conf = config.vectorstore

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=vs_conf.collection_name,
        persist_directory=vs_conf.persist_dir,  # 绝对路径：RagDemo/chroma_data
    )
    return vectorstore


def load_vectorstore() -> Chroma:
    """
    从本地加载已存在的 Chroma 向量库。
    """
    embeddings = create_embedding_model()
    vs_conf = config.vectorstore

    return Chroma(
        collection_name=vs_conf.collection_name,
        embedding_function=embeddings,
        persist_directory=vs_conf.persist_dir,  # 绝对路径：RagDemo/chroma_data
    )