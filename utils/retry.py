"""重试工具.

支持两种独立的重试策略：
  1. 超时重试：单次 LLM 请求超过指定时间后取消并重试
  2. 网络错误重试：遇到 HTTP 错误码（400/429/5xx 等）时线性退避重试
两个计数器独立运作，不互相消耗。
"""

import asyncio
import logging
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    timeout: float = 200.0,
    timeout_retries: int = 1,
    network_retries: int = 2,
    initial_delay: float = 1.5,
) -> T:
    """带超时 + 网络错误重试的异步调用包装.

    执行流程:
      - 每次调用 fn() 时，用 asyncio.wait_for 施加超时限制。
      - 如果触发超时（asyncio.TimeoutError），消耗一次 timeout_retries 后重试。
      - 如果触发网络/HTTP 错误（400/429/5xx 等），消耗一次 network_retries，
        线性退避后重试。
      - 两种计数器独立：超时不消耗网络重试次数，反之亦然。
      - 计数器用尽或遇到不可重试的异常时直接抛出。

    Args:
        fn: 要执行的异步函数（无参数闭包）.
        timeout: 单次调用的超时秒数，0 表示不设超时.
        timeout_retries: 超时后的最大重试次数.
        network_retries: 网络/HTTP 错误后的最大重试次数.
        initial_delay: 网络错误重试的初始退避秒数.

    Returns:
        fn 的返回值.

    Raises:
        asyncio.TimeoutError: 超时重试用尽后抛出.
        Exception: 网络重试用尽或不可重试错误时抛出原始异常.
    """
    timeout_used = 0
    network_used = 0
    last_error: Exception | None = None

    while True:
        try:
            if timeout > 0:
                return await asyncio.wait_for(fn(), timeout=timeout)
            else:
                return await fn()

        except asyncio.TimeoutError:
            timeout_used += 1
            logger.warning(
                "[Retry] Request timed out (%.0fs). Timeout retry %d/%d",
                timeout, timeout_used, timeout_retries,
            )
            if timeout_used > timeout_retries:
                logger.error(
                    "[Retry] Timeout retries exhausted (%d), aborting.",
                    timeout_retries,
                )
                raise
            # 超时重试不需要退避，直接重来

        except Exception as error:
            last_error = error

            # 从异常中提取 HTTP 状态码
            status = extract_status(error)

            is_retryable = is_retryable_error(status)

            if not is_retryable:
                logger.error(
                    "[Retry] Non-retryable error (status=%s): %s",
                    status, error,
                )
                raise

            network_used += 1

            if network_used > network_retries:
                logger.error(
                    "[Retry] Network retries exhausted (%d), status=%s, aborting. Error: %s",
                    network_retries, status, error,
                )
                raise

            # Linear backoff: initial_delay, 2*initial_delay, 3*initial_delay...
            delay = initial_delay * network_used
            logger.warning(
                "[Retry] Network error (status=%s). Network retry %d/%d in %.1fs... Error: %s",
                status or "no_status_code",
                network_used,
                network_retries,
                delay,
                error,
            )
            await asyncio.sleep(delay)


def extract_status(error: Exception) -> int | None:
    """从异常对象中尽力提取 HTTP 状态码.

    Args:
        error: 异常对象.

    Returns:
        HTTP 状态码整数，提取不到则返回 None.
    """
    status = getattr(error, "status_code", None) or getattr(
        error, "status", None
    )
    if status is None and hasattr(error, "response"):
        status = getattr(error.response, "status_code", None)
    # google-genai SDK 有时将 status 放在 code 属性里
    if status is None:
        status = getattr(error, "code", None)
    return status


def is_retryable_error(status: int | None) -> bool:
    """判断给定的 HTTP 状态码（或无状态码的纯网络错误）是否值得重试.

    以下情况视为可重试：
      - status 为 None：纯网络错误（连接超时、DNS 解析失败等）
      - 400：错误请求（某些中转/代理会用 400 包裹临时错误）
      - 429：速率限制
      - 500/502/503/504 等：服务端错误

    Args:
        status: HTTP 状态码或 None.

    Returns:
        是否可重试.
    """
    if status is None:
        return True  # 纯网络错误
    if isinstance(status, int):
        if status == 400 or status == 429:
            return True
        if 500 <= status < 600:
            return True
    return False
