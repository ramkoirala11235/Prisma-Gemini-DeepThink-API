"""Prisma DeepThink API 数据模型.

所有 Pydantic 模型定义，包含业务模型和 OpenAI 兼容的请求/响应模型。
迁移自原项目 types.ts。
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# --- 推理引擎内部模型 ---

ThinkingLevel = Literal["minimal", "low", "medium", "high"]
AppState = Literal[
    "idle", "analyzing", "experts_working", "reviewing", "synthesizing", "completed"
]
CheckpointPhase = Literal["planning", "experts", "review", "synthesis"]
CheckpointStatus = Literal["running", "interrupted", "completed", "error"]
ExpertContextStatus = Literal["active", "iterated", "deleted"]


class ExpertConfig(BaseModel):
    """Manager 规划阶段输出的专家配置."""

    role: str
    description: str
    temperature: float
    prompt: str


class ExpertResult(BaseModel):
    """Expert 执行后的完整结果."""

    id: str = ""
    role: str = ""
    description: str = ""
    temperature: float = 1.0
    prompt: str = ""
    status: Literal["pending", "thinking", "completed", "error"] = "pending"
    content: str = ""
    thoughts: str = ""
    round: int = 1
    context_status: ExpertContextStatus = "active"
    context_note: str = ""


class AnalysisResult(BaseModel):
    """Manager 规划阶段的完整输出."""

    thought_process: str = ""
    experts: list[ExpertConfig] = Field(default_factory=list)


class ReviewExpertAction(BaseModel):
    """审查阶段对单个专家的动作决策."""

    target_expert_id: str = ""
    target_expert_role: str = ""
    action: Literal["keep", "iterate", "delete"] = "keep"
    reason: str = ""
    strict_prompt: str = ""
    improvement_suggestions: str = ""
    iterated_expert: ExpertConfig | None = None


class ReviewResult(BaseModel):
    """Manager 审查阶段的输出."""

    satisfied: bool = True
    review_critique: str = ""
    overall_rejection_reason: str = ""
    critique: str = ""
    next_round_strategy: str = ""
    refined_experts: list[ExpertConfig] = Field(default_factory=list)
    expert_actions: list[ReviewExpertAction] = Field(default_factory=list)
    round: int = 0


class DeepThinkConfig(BaseModel):
    """深度推理的配置参数.

    各阶段温度覆盖字段（planning_temperature / expert_temperature /
    review_temperature / synthesis_temperature）设为具体数值后，
    该阶段的温度会被强制锁定，忽略请求温度和 Manager 分配的温度。
    保持 None 则沿用原有行为。
    """

    planning_level: ThinkingLevel = "high"
    expert_level: ThinkingLevel = "high"
    synthesis_level: ThinkingLevel = "high"
    enable_recursive_loop: bool = True
    max_rounds: int = 2
    max_context_messages: int = 10  # 保留的历史消息条数上限
    # 各阶段温度覆盖（None = 不覆盖，沿用请求温度或 Manager 分配温度）
    planning_temperature: Optional[float] = None
    expert_temperature: Optional[float] = None
    review_temperature: Optional[float] = None
    synthesis_temperature: Optional[float] = None


class DeepThinkCheckpoint(BaseModel):
    """Persistent state for `!deepthink_continue <id>` resume flow."""

    schema_version: int = 1
    resume_id: str
    request_model: str = ""
    real_model: str = ""
    manager_model: str = ""
    synthesis_model: str = ""

    phase: CheckpointPhase = "planning"
    status: CheckpointStatus = "running"
    current_round: int = 1

    analysis: AnalysisResult | None = None
    experts: list[ExpertResult] = Field(default_factory=list)
    reviews: list[ReviewResult] = Field(default_factory=list)

    reasoning_content: str = ""
    output_content: str = ""

    error_message: str = ""
    started_at: int = Field(default_factory=lambda: int(time.time()))
    updated_at: int = Field(default_factory=lambda: int(time.time()))
    completed_at: int | None = None


# --- OpenAI 兼容 API 模型 ---


class ChatMessageContent(BaseModel):
    """OpenAI 格式的消息.

    content 支持纯字符串或 OpenAI 多部分数组格式:
    [{"type": "text", "text": "..."}, ...]
    """

    role: Literal["system", "user", "assistant"]
    content: Any  # str 或 list[dict]

    @property
    def text(self) -> str:
        """将 content 统一转为纯文本字符串."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            # 拼接所有 text 类型的部分，跳过图片等非文本部分
            parts = []
            for item in self.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "\n".join(parts)
        return str(self.content)

    @property
    def image_parts(self) -> list[dict]:
        """提取消息中的图片，转为 Gemini inlineData 格式.

        解析 OpenAI 格式的 image_url（data:mime;base64,xxx），
        转为 {"inline_data": {"mime_type": ..., "data": ...}} 字典。

        Returns:
            Gemini inlineData 格式的图片列表，无图片则为空列表.
        """
        parts: list[dict] = []
        if not isinstance(self.content, list):
            return parts
        for item in self.content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            url = item.get("image_url", {}).get("url", "")
            if not url.startswith("data:"):
                continue
            # 解析 data:image/png;base64,xxxxx
            header, _, b64data = url.partition(",")
            if not b64data:
                continue
            mime = "image/png"
            if ":" in header and ";" in header:
                mime = header.split(":")[1].split(";")[0]
            parts.append({
                "inline_data": {"mime_type": mime, "data": b64data}
            })
        return parts


class ChatCompletionRequest(BaseModel):
    """OpenAI 兼容的 /v1/chat/completions 请求体."""

    model: str = "gemini-3-flash-preview"
    messages: list[ChatMessageContent]
    stream: bool = False
    temperature: Optional[float] = None

    # Prisma 扩展参数（非 OpenAI 标准）
    prisma_config: Optional[DeepThinkConfig] = None


class ChatCompletionChoice(BaseModel):
    """非流式响应中的单个 choice."""

    index: int = 0
    message: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str = "stop"


class ChatCompletionUsage(BaseModel):
    """Token 使用统计（简化版）."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI 兼容的非流式响应."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChatCompletionChoice] = Field(default_factory=list)
    usage: ChatCompletionUsage = Field(default_factory=ChatCompletionUsage)


class ChatCompletionChunkDelta(BaseModel):
    """流式响应中单个 chunk 的 delta."""

    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    """流式响应中的单个 choice."""

    index: int = 0
    delta: ChatCompletionChunkDelta = Field(
        default_factory=ChatCompletionChunkDelta
    )
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """OpenAI 兼容的流式响应 chunk."""

    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChatCompletionChunkChoice] = Field(default_factory=list)
