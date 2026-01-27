from __future__ import annotations

import os
import logging
import logging.handlers
import platform
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, Any

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

import sys

# Determine resource roots for finding bundled files vs external config (PyInstaller compatible)
def _get_resource_root() -> Path:
    """Returns the directory containing the executable or the script."""
    if getattr(sys, 'frozen', False):
        # Running from a PyInstaller bundle. use the executable's parent directory
        return Path(sys.executable).parent
    # Running from source (3 levels up from app/core/config.py)
    return Path(__file__).resolve().parent.parent.parent

def _get_bundle_root() -> Path:
    """Returns the internal bundle root (e.g., _MEIPASS in onefile mode or the same as resource_root in onedir/source)."""
    if getattr(sys, 'frozen', False):
        # sys._MEIPASS is the temporary directory where PyInstaller unzips files
        return Path(sys._MEIPASS)
    # Running from source
    return Path(__file__).resolve().parent.parent.parent


def _get_plugins_root() -> Path:
    # Add plugins directory to path for importing plugin services
    # We import directly from service modules to avoid circular imports with routers
    if getattr(sys, 'frozen', False):
        # 1. Try Bundled path (preferred for bundled onefile/onedir)
        root_dir = _bundle_root / "plugins"
        
        if not root_dir.is_dir():
            # 2. Try path next to executable (for external plugins in onedir)
            root_dir = _resource_root / "plugins"
            
        if not root_dir.is_dir():
            # Path(sys.executable) is dist/local-cocoa-server/local-cocoa-server.exe
            root_dir = Path(sys.executable).parent.parent.parent / "plugins"
    else:
        root_dir = _resource_root / "plugins"

    if root_dir.is_dir() and str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    return root_dir

_resource_root = _get_resource_root()
_bundle_root = _get_bundle_root()
_plugins_root = _get_plugins_root()

print(f"[config] Resource root (external): {str(_resource_root)}")
print(f"[config] Bundle root (internal): {str(_bundle_root)}")

# Determine environment mode (e.g., 'dev', 'prod', 'test')
# Default to 'dev' if ENV system environment variable is not set
_env_mode = os.getenv("ENV", "dev")

# Construct list of .env files to load
# We load bundled ones first, then allow external ones to override them
_env_files = [
    str(_bundle_root / ".env"),
    str(_bundle_root / f".env.{_env_mode}"),
]

# Shared configuration for all settings classes to ensure consistent loading priority:
# 1. Existing system environment variables (highest priority)
# 2. Local CWD '.env.{mode}' and '.env' (if they exist, override bundled ones)
# 3. Bundled '.env.{mode}' (middle priority)
# 4. Bundled '.env' (lowest priority)
# Note: In Pydantic Settings v2, LATER files in the tuple override earlier ones.
_common_config = SettingsConfigDict(
    env_file=tuple(_env_files),
    env_file_encoding="utf-8",
    case_sensitive=False,
    extra="ignore",
    env_ignore_empty=True
)

class ServiceEndpoints(BaseSettings):
    """Service endpoints configuration."""
    model_config = _common_config
    
    llm_host: str = Field(alias="LOCAL_SERVICE_LLM_HOST")
    llm_port: int = Field(alias="LOCAL_SERVICE_LLM_PORT")
    embedding_host: str = Field(alias="LOCAL_SERVICE_EMBEDDING_HOST")
    embedding_port: int = Field(alias="LOCAL_SERVICE_EMBEDDING_PORT")
    rerank_host: str = Field(alias="LOCAL_SERVICE_RERANK_HOST")
    rerank_port: int = Field(alias="LOCAL_SERVICE_RERANK_PORT")
    vision_host: str = Field(alias="LOCAL_SERVICE_VISION_HOST")
    vision_port: int = Field(alias="LOCAL_SERVICE_VISION_PORT")
    transcribe_host: str = Field(alias="LOCAL_SERVICE_TRANSCRIBE_HOST")
    transcribe_port: int = Field(alias="LOCAL_SERVICE_TRANSCRIBE_PORT")

    @property
    def llm_url(self) -> str: return f"http://{self.llm_host}:{self.llm_port}"
    @property
    def embedding_url(self) -> str: return f"http://{self.embedding_host}:{self.embedding_port}"
    @property
    def rerank_url(self) -> str: return f"http://{self.rerank_host}:{self.rerank_port}"
    @property
    def vision_url(self) -> str: return f"http://{self.vision_host}:{self.vision_port}"
    @property
    def transcribe_url(self) -> str: return f"http://{self.transcribe_host}:{self.transcribe_port}"


class QdrantConfig(BaseSettings):
    """Qdrant vector database configuration."""
    model_config = _common_config
    
    data_path: str = Field(alias="LOCAL_QDRANT_DATA_PATH")
    collection_name: str = Field(alias="LOCAL_QDRANT_COLLECTION_NAME")
    embedding_dim: int = Field(alias="LOCAL_QDRANT_EMBEDDING_DIM")
    metric_type: Literal["COSINE", "DOT", "EUCLID"] = Field(alias="LOCAL_QDRANT_METRIC_TYPE")

class Settings(BaseSettings):
    env: str = _env_mode
    is_win: bool = platform.system() == "Windows"

    """Main application settings with automatic .env file loading."""
    model_config = _common_config

    # Path settings
    runtime_root: Path = Field(alias="LOCAL_RUNTIME_ROOT")
    resource_root: Path = _resource_root
    bundle_root: Path = _bundle_root
    plugins_root: Path = _plugins_root
    models_config_path: Path = Field(alias="LOCAL_MODELS_CONFIG_PATH")
    service_bin_path: Path = Field(alias="LOCAL_SERVICE_BIN_PATH")
    llama_server_path: Path = Field(alias="LOCAL_LLAMA_SERVER_PATH")
    whisper_server_path: Path = Field(alias="LOCAL_WHISPER_SERVER_PATH")
    model_root_path: Path = Field(alias="LOCAL_MODEL_ROOT_PATH")

    endpoints: ServiceEndpoints = Field(default_factory=ServiceEndpoints)
    main_host: str = Field(alias="LOCAL_SERVICE_MAIN_HOST")
    main_port: int = Field(alias="LOCAL_SERVICE_MAIN_PORT")
    poll_interval_seconds: int = Field(alias="LOCAL_RAG_POLL_INTERVAL_SECONDS")
    refresh_on_startup: bool = Field(alias="LOCAL_RAG_REFRESH_ON_STARTUP")
    db_name: str = Field(alias="LOCAL_RAG_DB_NAME")
    max_depth: int = Field(alias="LOCAL_RAG_MAX_DEPTH")
    follow_symlinks: bool = Field(alias="LOCAL_RAG_FOLLOW_SYMLINKS")
    reuse_embeddings: bool = Field(alias="LOCAL_RAG_REUSE_EMBEDDINGS")
    embed_batch_size: int = Field(ge=1, alias="LOCAL_RAG_EMBED_BATCH_SIZE")
    embed_batch_delay_ms: int = Field(ge=0, alias="LOCAL_RAG_EMBED_BATCH_DELAY_MS")
    vision_batch_delay_ms: int = Field(ge=0, alias="LOCAL_RAG_VISION_BATCH_DELAY_MS")
    embed_max_chars: int = Field(ge=256, alias="LOCAL_RAG_EMBED_MAX_CHARS")
    snapshot_interval_seconds: int = Field(alias="LOCAL_RAG_SNAPSHOT_INTERVAL_SECONDS")
    supported_modes: tuple[Literal["search", "qa"], ...] = ("search", "qa")
    
    # LLM settings
    llm_context_tokens: int = Field(ge=2048, alias="LOCAL_LLM_CONTEXT_TOKENS")
    llm_max_prompt_tokens: int = Field(ge=1024, alias="LOCAL_LLM_MAX_PROMPT_TOKENS")
    llm_chars_per_token: int = Field(ge=1, alias="LOCAL_LLM_CHARS_PER_TOKEN")

    # Vision and video settings
    vision_max_pixels: int = Field(alias="LOCAL_VISION_MAX_PIXELS")
    video_max_pixels: int = Field(alias="LOCAL_VIDEO_MAX_PIXELS")

    # Search settings
    search_result_limit: int = Field(alias="LOCAL_SEARCH_RESULT_LIMIT")
    qa_context_limit: int = Field(alias="LOCAL_QA_CONTEXT_LIMIT")
    max_snippet_length: int = Field(alias="LOCAL_MAX_SNIPPET_LENGTH")

    # Summary settings
    summary_max_tokens: int = Field(ge=32, alias="LOCAL_SUMMARY_MAX_TOKENS")
    summary_input_max_chars: int = Field(alias="LOCAL_SUMMARY_INPUT_MAX_CHARS")

    # PDF settings
    pdf_page_max_tokens: int = Field(ge=256, alias="LOCAL_PDF_PAGE_MAX_TOKENS")
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    pdf_mode: Literal["text", "vision"] = Field(alias="LOCAL_PDF_MODE")
    pdf_one_chunk_per_page: bool = Field(alias="LOCAL_PDF_ONE_CHUNK_PER_PAGE")

    # Chunking settings
    rag_chunk_size: int = Field(alias="LOCAL_RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(alias="LOCAL_RAG_CHUNK_OVERLAP")

    # Indexing mode
    default_indexing_mode: Literal["fast", "deep"] = Field(alias="LOCAL_DEFAULT_INDEXING_MODE")

    # Logging settings
    log_level: str = Field(alias="LOG_LEVEL")
    log_to_file: bool = Field(alias="LOCAL_SERVICE_LOG_TO_FILE")
    main_log_path: Optional[str] = Field(alias="LOCAL_MAIN_LOG_PATH")
    embed_log_path: Optional[str] = Field(alias="LOCAL_EMBED_LOG_PATH")
    rerank_log_path: Optional[str] = Field(alias="LOCAL_RERANK_LOG_PATH")
    vlm_log_path: Optional[str] = Field(alias="LOCAL_VLM_LOG_PATH")
    whisper_log_path: Optional[str] = Field(alias="LOCAL_WHISPER_LOG_PATH")

    # Memory extraction
    enable_memory_extraction: bool = Field(alias="LOCAL_ENABLE_MEMORY_EXTRACTION")
    memory_user_id: str = Field(alias="LOCAL_MEMORY_USER_ID")
    memory_extraction_stage: Literal["fast", "deep", "none"] = Field(alias="LOCAL_MEMORY_EXTRACTION_STAGE")
    memory_chunk_size: int = Field(alias="LOCAL_MEMORY_CHUNK_SIZE")
    
    # Debug settings
    debug_port: Optional[int] = Field(default=None, alias="LOCAL_SERVICE_DEBUG_PORT")
    debug_wait: bool = Field(default=False, alias="LOCAL_SERVICE_DEBUG_WAIT")

    # Model IDs
    active_model_id: str = Field(alias="LOCAL_ACTIVE_MODEL_ID")
    active_embedding_model_id: str = Field(alias="LOCAL_ACTIVE_EMBEDDING_MODEL_ID")
    active_reranker_model_id: str = Field(alias="LOCAL_ACTIVE_RERANKER_MODEL_ID")
    active_audio_model_id: str = Field(alias="LOCAL_ACTIVE_AUDIO_MODEL_ID")

    @property
    def db_path(self) -> Path:
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        return self.runtime_root / self.db_name

    @property
    def settings_path(self) -> Path:
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        return self.runtime_root / "rag_settings.json"

    @property
    def is_dev(self) -> bool:
        return self.env == "dev"

    @property
    def is_test(self) -> bool:
        return self.env == "test"

    @property
    def is_prod(self) -> bool:
        return self.env == "prod"

    
    @model_validator(mode='after')
    def resolve_all_placeholders(self) -> 'Settings':
        """
        Final pass to resolve any ${VAR} placeholders after all sources (env, .env, .env.mode) 
        have been merged. variables derived from another variable (like LOCAL_LLAMA_SERVER_PATH=<LOCAL_RUNTIME_ROOT>/llama-cpp/bin)
        must use this post validator to re-calculate derived values when parent values is ready
        """
        runtime_root_str = str(self.runtime_root)
        service_bin_path_str = str(self.service_bin_path)
        main_host_str = str(self.main_host)
        
        # Define replacement map
        replacements = {
            "<LOCAL_RUNTIME_ROOT>": runtime_root_str,
            "<LOCAL_SERVICE_BIN_PATH>": service_bin_path_str,
            "<LOCAL_SERVICE_MAIN_HOST>": main_host_str,
            # Add other base variables here if needed
        }

        def _resolve_str(v: Any) -> Any:
            if isinstance(v, str):
                for placeholder, replacement in replacements.items():
                    if placeholder in v:
                        v = v.replace(placeholder, replacement)
            return v

        # 1. Resolve top-level string fields
        self.llama_server_path = Path(_resolve_str(str(self.llama_server_path)))
        self.whisper_server_path = Path(_resolve_str(str(self.whisper_server_path)))
        self.model_root_path = Path(_resolve_str(str(self.model_root_path)))

        self.endpoints.llm_host = _resolve_str(self.endpoints.llm_host)
        self.endpoints.embedding_host = _resolve_str(self.endpoints.embedding_host)
        self.endpoints.rerank_host = _resolve_str(self.endpoints.rerank_host)
        self.endpoints.vision_host = _resolve_str(self.endpoints.vision_host)
        self.endpoints.transcribe_host = _resolve_str(self.endpoints.transcribe_host)

        # 2. Resolve nested configs
        self.qdrant.data_path = _resolve_str(self.qdrant.data_path)
        
        # 3. Resolve logging paths
        if self.main_log_path: self.main_log_path = _resolve_str(self.main_log_path)
        if self.embed_log_path: self.embed_log_path = _resolve_str(self.embed_log_path)
        if self.rerank_log_path: self.rerank_log_path = _resolve_str(self.rerank_log_path)
        if self.vlm_log_path: self.vlm_log_path = _resolve_str(self.vlm_log_path)
        if self.whisper_log_path: self.whisper_log_path = _resolve_str(self.whisper_log_path)

        return self

    def save_to_file(self):
        import json
        data = {
            "vision_max_pixels": self.vision_max_pixels,
            "video_max_pixels": self.video_max_pixels,
            "embed_batch_size": self.embed_batch_size,
            "embed_batch_delay_ms": self.embed_batch_delay_ms,
            "vision_batch_delay_ms": self.vision_batch_delay_ms,
            "search_result_limit": self.search_result_limit,
            "qa_context_limit": self.qa_context_limit,
            "max_snippet_length": self.max_snippet_length,
            "summary_max_tokens": self.summary_max_tokens,
            "pdf_one_chunk_per_page": self.pdf_one_chunk_per_page,
            "rag_chunk_size": self.rag_chunk_size,
            "rag_chunk_overlap": self.rag_chunk_overlap,
            "default_indexing_mode": self.default_indexing_mode,
            "enable_memory_extraction": self.enable_memory_extraction,
            "memory_extraction_stage": self.memory_extraction_stage,
            "memory_chunk_size": self.memory_chunk_size,
            "active_model_id": self.active_model_id,
            "active_embedding_model_id": self.active_embedding_model_id,
            "active_reranker_model_id": self.active_reranker_model_id,
            "active_audio_model_id": self.active_audio_model_id,
            "llm_context_tokens": self.llm_context_tokens,
        }
        try:
            with open(self.settings_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to save settings: {e}")

    def load_from_file(self):
        import json
        if not self.settings_path.exists():
            return
        try:
            with open(self.settings_path, 'r') as f:
                data = json.load(f)
                for key, value in data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to load settings: {e}")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.load_from_file()
    return s


settings = get_settings()
