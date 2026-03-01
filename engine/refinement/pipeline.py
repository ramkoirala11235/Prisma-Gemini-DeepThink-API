"""精修主流水线.

编排所有精修阶段: 规划 -> 专家执行 -> 规范审核 -> 初稿 -> 审查 ->
改进专家 -> 综合合并 -> 应用精修 -> 迭代或输出.
通过 asyncio.Queue 推送状态和最终输出, 与 orchestrator 对接.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

from config import get_thinking_budget, LLM_NETWORK_RETRIES
from engine.refinement import (
    applier,
    compliance,
    draft,
    improver,
    merger,
    planner,
    reviewer,
)
from models import (
    DeepThinkCheckpoint,
    DeepThinkConfig,
    DiffOperation,
    RefinementExpertConfig,
)
from prompts import (
    MSG_REFINEMENT_APPLIED,
    MSG_REFINEMENT_COMPLIANCE_CHECK,
    MSG_REFINEMENT_COMPLIANCE_FAILED,
    MSG_REFINEMENT_COMPLIANCE_PASSED,
    MSG_REFINEMENT_DRAFT_DONE,
    MSG_REFINEMENT_DRAFT_START,
    MSG_REFINEMENT_EXPERT_DONE,
    MSG_REFINEMENT_EXPERT_START,
    MSG_REFINEMENT_IMPROVER_DONE,
    MSG_REFINEMENT_IMPROVER_START,
    MSG_REFINEMENT_MERGE_DONE,
    MSG_REFINEMENT_MERGE_START,
    MSG_REFINEMENT_NEXT_ROUND,
    MSG_REFINEMENT_OUTPUT,
    MSG_REFINEMENT_PLANNING,
    MSG_REFINEMENT_REVIEW_APPROVED,
    MSG_REFINEMENT_REVIEW_START,
    MSG_PIPELINE_START,
    build_refinement_expert_contents,
    format_expert_task,
    get_refinement_expert_system_instruction,
)
from clients.llm_client import generate_content
from utils.retry import extract_status, is_retryable_error

logger = logging.getLogger(__name__)


def _now_ts() -> int:
    return int(time.time())


async def _run_single_expert(
    model: str,
    expert_cfg: RefinementExpertConfig,
    query: str,
    context: str,
    budget: int,
    user_system_prompt: str = "",
    image_parts: list[dict] | None = None,
    provider: str = "",
    forced_temperature: float | None = None,
) -> dict[str, str]:
    """执行单个精修专家, 返回 {role, domain, content}.

    使用 prefill 确认轮提高执行力.
    """
    system_instruction = get_refinement_expert_system_instruction(
        role=expert_cfg.role,
        domain=expert_cfg.domain,
        context=context,
        all_expert_roles=expert_cfg.all_expert_roles,
        user_system_prompt=user_system_prompt,
    )

    task_prompt = format_expert_task(query, expert_cfg.prompt)
    contents = build_refinement_expert_contents(
        task_prompt, image_parts=image_parts,
    )

    temperature = (
        forced_temperature
        if forced_temperature is not None
        else expert_cfg.temperature
    )

    max_retries = LLM_NETWORK_RETRIES

    for attempt in range(max_retries + 1):
        try:
            full_content, _, _ = await generate_content(
                model=model,
                contents=contents,
                system_instruction=system_instruction,
                temperature=temperature,
                thinking_budget=budget,
                provider=provider,
            )

            if not full_content.strip():
                if attempt < max_retries:
                    delay = 1.5 * (attempt + 1)
                    logger.warning(
                        "[RefinementExpert] %s empty response, retry %d/%d",
                        expert_cfg.role, attempt + 1, max_retries,
                    )
                    await asyncio.sleep(delay)
                    continue
                full_content = "（专家未生成有效内容）"

            return {
                "role": expert_cfg.role,
                "domain": expert_cfg.domain,
                "content": full_content,
            }

        except Exception as e:
            status = extract_status(e)
            retryable = is_retryable_error(status)
            if retryable and attempt < max_retries:
                delay = 1.5 * (attempt + 1)
                logger.warning(
                    "[RefinementExpert] %s error (status=%s), retry %d/%d: %s",
                    expert_cfg.role, status, attempt + 1, max_retries, e,
                )
                await asyncio.sleep(delay)
                continue

            logger.error(
                "[RefinementExpert] %s failed: %s", expert_cfg.role, e,
            )
            return {
                "role": expert_cfg.role,
                "domain": expert_cfg.domain,
                "content": f"（专家执行失败: {e}）",
            }

    return {
        "role": expert_cfg.role,
        "domain": expert_cfg.domain,
        "content": "（专家重试次数耗尽）",
    }


async def run_refinement_pipeline(
    queue: asyncio.Queue,
    query: str,
    history: list[dict[str, str]],
    model: str,
    mgr_model: str,
    syn_model: str,
    config: DeepThinkConfig,
    temperature: Optional[float],
    system_prompt: str = "",
    image_parts: list[dict] | None = None,
    resume_checkpoint: DeepThinkCheckpoint | None = None,
    provider: str = "",
) -> None:
    """精修流水线主入口.

    Args:
        queue: 输出 queue, 推送 (text, thought, phase, grounding) 元组.
        query: 用户原始问题.
        history: 对话历史.
        model: Expert 模型.
        mgr_model: Manager/规划 模型.
        syn_model: Synthesis 模型.
        config: 配置参数.
        temperature: 默认温度.
        system_prompt: 用户 system prompt.
        image_parts: 图片列表.
        resume_checkpoint: 断点恢复数据.
        provider: provider 标识符.
    """

    async def _emit(text: str) -> None:
        """推送思维链状态文本."""
        await queue.put(("", f"{text}\n", "refinement", []))

    def _set_refinement_phase(phase: str) -> None:
        """更新 checkpoint 的精修阶段标记."""
        if resume_checkpoint:
            resume_checkpoint.refinement_phase = phase
            resume_checkpoint.updated_at = _now_ts()

    # 计算各阶段预算
    planning_budget = get_thinking_budget(config.planning_level, model)
    expert_budget = get_thinking_budget(config.expert_level, model)
    synthesis_budget = get_thinking_budget(config.synthesis_level, model)

    # 获取精修专用配置
    compliance_model = config.compliance_model or "gemini-3-flash-preview"
    draft_model = config.draft_model or model
    review_model = config.review_model or mgr_model
    merge_model = config.merge_model or syn_model
    json_repair_model = config.json_repair_model or "gemini-3-flash-preview"
    max_refinement_rounds = config.refinement_max_rounds
    max_compliance_retries = config.compliance_check_max_retries
    enable_json_repair = config.enable_json_repair

    # 对话上下文
    max_ctx = config.max_context_messages
    recent_history = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history[-max_ctx:]
    )

    # --- 断点恢复: 确定起始阶段 ---
    start_phase = "planning"
    approved_outputs: list[dict[str, str]] = []
    draft_text = ""
    previous_merge_summary = ""
    start_round = 1

    if resume_checkpoint and resume_checkpoint.pipeline_mode == "refinement":
        start_phase = resume_checkpoint.refinement_phase or "planning"
        # 恢复已保存的中间状态
        if resume_checkpoint.refinement_expert_outputs:
            approved_outputs = list(resume_checkpoint.refinement_expert_outputs)
        if resume_checkpoint.draft_content:
            draft_text = resume_checkpoint.draft_content
        if resume_checkpoint.refinement_merge_summary:
            previous_merge_summary = resume_checkpoint.refinement_merge_summary
        if resume_checkpoint.refinement_round > 0:
            start_round = resume_checkpoint.refinement_round

        logger.info(
            "[RefinementPipeline] resuming from phase=%s round=%d "
            "experts=%d draft_len=%d",
            start_phase, start_round,
            len(approved_outputs), len(draft_text),
        )

    # 阶段顺序用于判断是否需要跳过
    _PHASE_ORDER = [
        "planning", "experts", "draft", "review",
        "improvers", "merge", "apply", "output",
    ]

    def _should_skip(phase: str) -> bool:
        """判断当前阶段是否已在 checkpoint 中完成, 应跳过."""
        if start_phase == "planning":
            return False
        try:
            return _PHASE_ORDER.index(phase) < _PHASE_ORDER.index(start_phase)
        except ValueError:
            return False

    try:
        # ========== 阶段 1: 规划 ==========
        if not _should_skip("planning"):
            await _emit(MSG_PIPELINE_START)
            await asyncio.sleep(2)

        if not _should_skip("experts"):
            _set_refinement_phase("planning")
            await _emit(MSG_REFINEMENT_PLANNING)

            expert_configs = await planner.plan(
                model=mgr_model,
                query=query,
                context=recent_history,
                budget=planning_budget,
                temperature=(
                    config.planning_temperature
                    if config.planning_temperature is not None
                    else temperature
                ),
                user_system_prompt=system_prompt,
                image_parts=image_parts,
                provider=provider,
            )

            if not expert_configs:
                # 兜底: 分配一个通用专家
                expert_configs = [RefinementExpertConfig(
                    role="通用分析专家",
                    domain="全面分析用户需求",
                    prompt=query,
                    all_expert_roles=["通用分析专家"],
                )]

            expert_names = "、".join(e.role for e in expert_configs)
            await _emit(f"已分配 {len(expert_configs)} 位专家：{expert_names}")

            # ========== 阶段 2+3: 专家并行执行 + 即时规范审核 ==========
            _set_refinement_phase("experts")

            for ec in expert_configs:
                await _emit(
                    MSG_REFINEMENT_EXPERT_START.format(
                        expert_name=ec.role, domain=ec.domain,
                    )
                )

            forced_expert_temp = config.expert_temperature

            async def _run_expert_with_compliance(
                ec: RefinementExpertConfig,
            ) -> dict[str, str]:
                """单个专家: 执行 -> 即时审核 -> 不过则重试."""
                eo = await _run_single_expert(
                    model=model,
                    expert_cfg=ec,
                    query=query,
                    context=recent_history,
                    budget=expert_budget,
                    user_system_prompt=system_prompt,
                    image_parts=image_parts,
                    provider=provider,
                    forced_temperature=forced_expert_temp,
                )
                await _emit(MSG_REFINEMENT_EXPERT_DONE.format(expert_name=eo["role"]))
                if eo["content"]:
                    await queue.put(
                        ("", f"\n```content\n{eo['content']}\n```\n", "experts", [])
                    )

                # 即时规范审核
                await _emit(
                    MSG_REFINEMENT_COMPLIANCE_CHECK.format(expert_name=eo["role"])
                )

                for retry in range(max_compliance_retries + 1):
                    check_result = await compliance.check_compliance(
                        content=eo["content"],
                        role=eo["role"],
                        domain=eo.get("domain", ""),
                        task=ec.prompt,
                        model=compliance_model,
                        provider=provider,
                        enable_json_repair=enable_json_repair,
                        json_repair_model=json_repair_model,
                    )

                    if check_result.passed:
                        await _emit(
                            MSG_REFINEMENT_COMPLIANCE_PASSED.format(
                                expert_name=eo["role"],
                            )
                        )
                        return eo

                    if retry < max_compliance_retries:
                        await _emit(
                            MSG_REFINEMENT_COMPLIANCE_FAILED.format(
                                expert_name=eo["role"],
                                reason=check_result.reason[:200],
                            )
                        )
                        # 重新生成
                        eo = await _run_single_expert(
                            model=model,
                            expert_cfg=ec,
                            query=query,
                            context=recent_history,
                            budget=expert_budget,
                            user_system_prompt=system_prompt,
                            image_parts=image_parts,
                            provider=provider,
                            forced_temperature=forced_expert_temp,
                        )
                        if eo["content"]:
                            await queue.put(
                                ("", f"\n```content\n{eo['content']}\n```\n", "experts", [])
                            )
                    else:
                        # 超过重试次数, 强制放行
                        await _emit(
                            MSG_REFINEMENT_COMPLIANCE_PASSED.format(
                                expert_name=eo["role"],
                            )
                        )

                return eo

            approved_outputs = list(await asyncio.gather(
                *[_run_expert_with_compliance(ec) for ec in expert_configs]
            ))

            if not approved_outputs:
                approved_outputs = [{"role": "fallback", "domain": "", "content": query}]

            # 保存精修专家输出到 checkpoint
            if resume_checkpoint:
                resume_checkpoint.refinement_expert_outputs = [
                    {"role": o["role"], "domain": o.get("domain", ""), "content": o["content"]}
                    for o in approved_outputs
                ]
                resume_checkpoint.updated_at = _now_ts()

        # ========== 阶段 4: 初稿生成 ==========
        if not _should_skip("draft"):
            _set_refinement_phase("draft")
            await _emit(MSG_REFINEMENT_DRAFT_START)

            draft_text = await draft.generate_draft(
                model=draft_model,
                query=query,
                context=recent_history,
                expert_outputs=approved_outputs,
                budget=synthesis_budget,
                temperature=(
                    config.synthesis_temperature
                    if config.synthesis_temperature is not None
                    else temperature
                ),
                user_system_prompt=system_prompt,
                image_parts=image_parts,
                provider=provider,
            )

            await _emit(MSG_REFINEMENT_DRAFT_DONE)
            await queue.put(("", f"\n```content\n{draft_text}\n```\n", "draft", []))

            # 保存初稿到 checkpoint
            if resume_checkpoint:
                resume_checkpoint.draft_content = draft_text
                resume_checkpoint.updated_at = _now_ts()

        # ========== 阶段 5-8: 精修迭代循环 ==========
        global_op_id_counter = 0

        for refinement_round in range(start_round, max_refinement_rounds + 1):
            # 更新 checkpoint
            if resume_checkpoint:
                resume_checkpoint.refinement_round = refinement_round
                resume_checkpoint.updated_at = _now_ts()

            # --- 5. 审查 ---
            if not (_should_skip("review") and refinement_round == start_round):
                _set_refinement_phase("review")
                await _emit(
                    MSG_REFINEMENT_REVIEW_START.format(round=refinement_round)
                )

                review_analysis = await reviewer.review_draft(
                    model=review_model,
                    query=query,
                    draft_text=draft_text,
                    budget=planning_budget,
                    refinement_round=refinement_round,
                    previous_summary=previous_merge_summary,
                    temperature=(
                        config.review_temperature
                        if config.review_temperature is not None
                        else 0.7
                    ),
                    user_system_prompt=system_prompt,
                    image_parts=image_parts,
                    provider=provider,
                    enable_json_repair=enable_json_repair,
                    json_repair_model=json_repair_model,
                )

                # 迭代轮 (>= 2) 允许通过
                if review_analysis.approved and refinement_round >= 2:
                    await _emit(MSG_REFINEMENT_REVIEW_APPROVED)
                    break

                if not review_analysis.refinement_experts:
                    await _emit(MSG_REFINEMENT_REVIEW_APPROVED)
                    break

                # 输出审查发现的问题
                if review_analysis.issues:
                    issues_text = "\n".join(
                        f"- {issue}" for issue in review_analysis.issues
                    )
                    await _emit(f"审查发现的问题：\n{issues_text}")

            # --- 6. 改进专家并行执行 ---
            if not (_should_skip("improvers") and refinement_round == start_round):
                _set_refinement_phase("improvers")

                draft_lines_json = json.dumps(
                    reviewer.split_draft_to_lines(draft_text),
                    ensure_ascii=False,
                )

                improver_configs = review_analysis.refinement_experts
                improver_names = "、".join(e.role for e in improver_configs)
                await _emit(f"已分配 {len(improver_configs)} 位改进专家：{improver_names}")

                for ic in improver_configs:
                    await _emit(
                        MSG_REFINEMENT_IMPROVER_START.format(
                            expert_name=ic.role, domain=ic.domain,
                        )
                    )

                improver_tasks = [
                    improver.run_improver(
                        model=model,
                        expert_config=ic,
                        draft_lines_json=draft_lines_json,
                        budget=expert_budget,
                        guidance=review_analysis.expert_guidance.get(ic.role, ""),
                        user_system_prompt=system_prompt,
                        image_parts=image_parts,
                        provider=provider,
                        enable_json_repair=enable_json_repair,
                        json_repair_model=json_repair_model,
                    )
                    for ic in improver_configs
                ]
                improver_results = await asyncio.gather(*improver_tasks)

                # 合并所有操作并分配全局 op_id
                all_operations: list[DiffOperation] = []
                for ir in improver_results:
                    await _emit(
                        MSG_REFINEMENT_IMPROVER_DONE.format(
                            expert_name=ir.role,
                            op_count=len(ir.operations),
                        )
                    )
                    # 推送改进专家的分析到思维链（换行 + 代码块包裹）
                    if ir.analysis:
                        await _emit(
                            f"「{ir.role}」分析：\n```content\n{ir.analysis[:5000]}\n```"
                        )

                    for op in ir.operations:
                        op.op_id = global_op_id_counter
                        op.expert_role = ir.role
                        all_operations.append(op)
                        global_op_id_counter += 1

                # 保存改进专家结果到 checkpoint
                if resume_checkpoint:
                    resume_checkpoint.refinement_improver_results = [
                        {
                            "role": ir.role,
                            "analysis": ir.analysis,
                            "operations": [
                                op.model_dump(mode="json") for op in ir.operations
                            ],
                        }
                        for ir in improver_results
                    ]
                    resume_checkpoint.updated_at = _now_ts()

                if not all_operations:
                    await _emit("改进专家未提交任何操作, 跳过合并。")
                    break

            # --- 7. 综合助手合并 ---
            if not (_should_skip("merge") and refinement_round == start_round):
                _set_refinement_phase("merge")
                await _emit(MSG_REFINEMENT_MERGE_START)

                merge_result = await merger.merge_operations(
                    model=merge_model,
                    draft_text=draft_text,
                    operations=all_operations,
                    budget=synthesis_budget,
                    temperature=(
                        config.synthesis_temperature
                        if config.synthesis_temperature is not None
                        else 0.5
                    ),
                    provider=provider,
                    enable_json_repair=enable_json_repair,
                    json_repair_model=json_repair_model,
                )

                accepted = sum(
                    1 for d in merge_result.decisions if d.decision == "accept"
                )
                rejected = sum(
                    1 for d in merge_result.decisions if d.decision == "reject"
                )
                modified = sum(
                    1 for d in merge_result.decisions if d.decision == "modify"
                )
                await _emit(
                    MSG_REFINEMENT_MERGE_DONE.format(
                        accepted=accepted, rejected=rejected, modified=modified,
                    )
                )
                if merge_result.summary:
                    await _emit(
                        f"综合简评：\n```content\n{merge_result.summary}\n```"
                    )

                previous_merge_summary = merge_result.summary

                # 保存综合摘要到 checkpoint
                if resume_checkpoint:
                    resume_checkpoint.refinement_merge_summary = previous_merge_summary
                    resume_checkpoint.updated_at = _now_ts()

            # --- 8. 应用精修 ---
            _set_refinement_phase("apply")
            draft_text = applier.apply_refinements(
                draft_text, all_operations, merge_result.decisions,
            )
            await _emit(MSG_REFINEMENT_APPLIED)

            # 更新 checkpoint
            if resume_checkpoint:
                resume_checkpoint.draft_content = draft_text
                resume_checkpoint.updated_at = _now_ts()

            # 判断是否继续迭代
            remaining = max_refinement_rounds - refinement_round
            if remaining <= 0:
                break

            await _emit(
                MSG_REFINEMENT_NEXT_ROUND.format(round=refinement_round + 1)
            )

        # ========== 阶段 9: 输出最终结果 ==========
        _set_refinement_phase("output")
        await _emit(MSG_REFINEMENT_OUTPUT)

        # 流式输出精修后的最终文本
        # 分块推送, 模拟流式效果
        chunk_size = 200
        for i in range(0, len(draft_text), chunk_size):
            chunk = draft_text[i:i + chunk_size]
            await queue.put((chunk, "", "synthesis", []))
            await asyncio.sleep(0.01)  # 微小延迟让 SSE 更流畅

    except asyncio.CancelledError:
        logger.info("[RefinementPipeline] cancelled")
        raise
    except Exception as exc:
        logger.exception("[RefinementPipeline] failed")
        from prompts import REFINEMENT_FALLBACK_TEXT
        await queue.put((REFINEMENT_FALLBACK_TEXT, "", "system_error", []))
