from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # VLM API (multimodal Q&A and captioning).
    # Two formats are supported:
    #   - "responses": Doubao / Volcano Ark Responses API
    #     (POST {base_url}/responses with input=[{role, content:[{type:input_image|input_text}]}])
    #   - "chat_completions": OpenAI-compatible Chat Completions
    #     (POST {base_url}/chat/completions, used by Xiaomi MiMo and others)
    vlm_api_provider: str = "doubao"
    vlm_api_format: Literal["responses", "chat_completions"] = "responses"
    vlm_api_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    vlm_api_key: str = ""
    vlm_model_name: str = "doubao-seed-2-0-pro-260215"
    vlm_api_timeout: int = 120

    # Distillation / research backbone split. The student VLM can be switched to
    # a local OpenAI-compatible endpoint without changing the text orchestrator.
    agent_vlm_backend: Literal["remote", "local"] = "remote"
    local_vlm_base_url: str = ""
    local_vlm_model_name: str = ""
    local_vlm_api_key: str = "EMPTY"

    # Tool-aware text orchestrator. It must speak an OpenAI-compatible
    # Chat Completions API. Empty values default to the VLM API settings above.
    orchestrator_api_base_url: str = ""
    orchestrator_api_key: str = ""
    orchestrator_model_name: str = ""
    orchestrator_api_timeout: int | None = None
    orchestrator_temperature: float = 0.2
    orchestrator_max_tool_calls: int = 8
    # Some OpenAI-compatible proxies refuse streaming for certain models
    # (e.g. "没有可用的 Provider 支持模型 X 的流式请求"). Default off so eval
    # works against the widest set of providers; UX path doesn't depend on
    # token-by-token streaming from the orchestrator.
    orchestrator_streaming: bool = False

    # Multimodal payload limits. Static charts/tables/images are passed through
    # the same "frames" interface used by the local Qwen3-VL servers.
    vqa_max_frames: int = 6
    vqa_max_image_side: int = 448
    vqa_image_quality: int = 75
    vqa_max_output_tokens: int = 1024
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
