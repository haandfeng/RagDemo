# app/llm_client.py
from typing import Literal
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from .config import config


ProviderType = Literal["siliconflow"]


@dataclass
class LLMFactory:
    provider: ProviderType = "siliconflow"

    def create_chat_model(self) -> BaseChatModel:
        if self.provider != "siliconflow":
            raise ValueError(f"未知 provider: {self.provider}")

        sf = config.siliconflow
        if not sf.api_key:
            raise ValueError("SiliconFlow api_key 未配置，请检查 config.yaml")

        return ChatOpenAI(
            api_key=sf.api_key,
            base_url=sf.base_url,     # https://api.siliconflow.cn/v1
            model=sf.chat_model,      # Qwen/Qwen3-8B
            temperature=0.2,
        )