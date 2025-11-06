# app/config.py
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

# 项目根目录：RagDemo
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 默认配置文件路径：RagDemo/config.yaml
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class SiliconFlowConfig:
    api_key: str
    base_url: str = "https://api.siliconflow.cn/v1"
    chat_model: str = "Qwen/Qwen3-8B"
    embedding_model: str = "BAAI/bge-m3"


@dataclass
class VectorStoreConfig:
    # 这里会存成「绝对路径」，指向 RagDemo/chroma_data
    persist_dir: str
    collection_name: str = "documents"


@dataclass
class AppConfig:
    siliconflow: SiliconFlowConfig
    vectorstore: VectorStoreConfig
    # 这里也会存成「绝对路径」，指向 RagDemo/prompts/rag_prompts.yaml
    prompt_file: str


def _load_raw_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件格式错误，应为 YAML 映射对象: {path}")
    return data


def _build_app_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    raw = _load_raw_config(path)

    sf_raw: Dict[str, Any] = raw.get("siliconflow", {}) or {}
    vs_raw: Dict[str, Any] = raw.get("vectorstore", {}) or {}
    prompt_file_raw: str = raw.get("prompt_file", "prompts/rag_prompts.yaml")

    # ---- 处理向量库目录：相对路径 -> 相对于项目根目录 ----
    persist_dir_raw = vs_raw.get("persist_dir", "chroma_data")
    persist_path = Path(persist_dir_raw)
    if not persist_path.is_absolute():
        persist_path = PROJECT_ROOT / persist_path

    siliconflow = SiliconFlowConfig(
        api_key=sf_raw.get("api_key", ""),
        base_url=sf_raw.get("base_url", "https://api.siliconflow.cn/v1"),
        chat_model=sf_raw.get("chat_model", "Qwen/Qwen3-8B"),
        embedding_model=sf_raw.get("embedding_model", "BAAI/bge-m3"),
    )

    vectorstore = VectorStoreConfig(
        persist_dir=str(persist_path),
        collection_name=vs_raw.get("collection_name", "documents"),
    )

    # ---- 处理 prompt 文件路径：相对路径 -> 相对于项目根目录 ----
    prompt_path = Path(prompt_file_raw)
    if not prompt_path.is_absolute():
        prompt_path = PROJECT_ROOT / prompt_path

    return AppConfig(
        siliconflow=siliconflow,
        vectorstore=vectorstore,
        prompt_file=str(prompt_path),
    )


config = _build_app_config()