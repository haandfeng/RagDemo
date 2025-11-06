# app/embeddings.py
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

from .config import config


def create_embedding_model() -> Embeddings:
    sf = config.siliconflow
    if not sf.api_key:
        raise ValueError("SiliconFlow api_key 未配置，请检查 config.yaml")

    return OpenAIEmbeddings(
        api_key=sf.api_key,
        base_url=sf.base_url,        # https://api.siliconflow.cn/v1
        model=sf.embedding_model,    # BAAI/bge-m3
    )