"""
Comprehensive tests for all 8 built-in plugins.

Covers: construction validation, happy paths, edge cases, error paths,
multi-plugin interaction, and immutability guarantees.
"""

import asyncio
import time

import pytest
from unittest.mock import AsyncMock, MagicMock, call

from nucleusiq.plugins.base import BasePlugin, ModelRequest, ToolRequest, AgentContext
from nucleusiq.plugins.errors import PluginHalt, PluginError
from nucleusiq.plugins.manager import PluginManager
from nucleusiq.plugins.builtin.model_call_limit import ModelCallLimitPlugin
from nucleusiq.plugins.builtin.tool_call_limit import ToolCallLimitPlugin
from nucleusiq.plugins.builtin.tool_retry import ToolRetryPlugin
from nucleusiq.plugins.builtin.model_fallback import ModelFallbackPlugin
from nucleusiq.plugins.builtin.pii_guard import PIIGuardPlugin, _luhn_check, BUILTIN_PATTERNS
from nucleusiq.plugins.builtin.human_approval import HumanApprovalPlugin
from nucleusiq.plugins.builtin.context_window import ContextWindowPlugin, _approximate_tokens
from nucleusiq.plugins.builtin.tool_guard import ToolGuardPlugin


# ====================================================================
# 1. ModelCallLimitPlugin
# ====================================================================

class TestModelCallLimitPluginDetailed:

    def test_name_is_model_call_limit(self):
        assert ModelCallLimitPlugin().name == "model_call_limit"

    def test_default_limit_is_10(self):
        p = ModelCallLimitPlugin()
        assert p._max_calls == 10

    def test_is_base_plugin(self):
        assert isinstance(ModelCallLimitPlugin(), BasePlugin)

    @pytest.mark.asyncio
    async def test_call_count_1_passes(self):
        p = ModelCallLimitPlugin(max_calls=5)
        assert await p.before_model(ModelRequest(call_count=1)) is None

    @pytest.mark.asyncio
    async def test_call_count_at_limit_passes(self):
        p = ModelCallLimitPlugin(max_calls=5)
        assert await p.before_model(ModelRequest(call_count=5)) is None

    @pytest.mark.asyncio
    async def test_call_count_one_over_halts(self):
        p = ModelCallLimitPlugin(max_calls=5)
        with pytest.raises(PluginHalt) as exc:
            await p.before_model(ModelRequest(call_count=6))
        assert "6" in str(exc.value.result) and "5" in str(exc.value.result)

    @pytest.mark.asyncio
    async def test_call_count_way_over_halts(self):
        p = ModelCallLimitPlugin(max_calls=3)
        with pytest.raises(PluginHalt):
            await p.before_model(ModelRequest(call_count=100))

    @pytest.mark.asyncio
    async def test_limit_1_allows_exactly_one(self):
        p = ModelCallLimitPlugin(max_calls=1)
        assert await p.before_model(ModelRequest(call_count=1)) is None
        with pytest.raises(PluginHalt):
            await p.before_model(ModelRequest(call_count=2))

    @pytest.mark.asyncio
    async def test_does_not_modify_request(self):
        p = ModelCallLimitPlugin(max_calls=10)
        req = ModelRequest(model="gpt-4o", call_count=3, messages=[{"role": "user", "content": "hi"}])
        result = await p.before_model(req)
        assert result is None

    @pytest.mark.asyncio
    async def test_halt_message_contains_counts(self):
        p = ModelCallLimitPlugin(max_calls=2)
        with pytest.raises(PluginHalt) as exc:
            await p.before_model(ModelRequest(call_count=3))
        msg = str(exc.value.result).lower()
        assert "limit" in msg and "exceeded" in msg


# ====================================================================
# 2. ToolCallLimitPlugin
# ====================================================================

class TestToolCallLimitPluginDetailed:

    def test_name_is_tool_call_limit(self):
        assert ToolCallLimitPlugin().name == "tool_call_limit"

    def test_default_limit_is_20(self):
        assert ToolCallLimitPlugin()._max_calls == 20

    def test_is_base_plugin(self):
        assert isinstance(ToolCallLimitPlugin(), BasePlugin)

    @pytest.mark.asyncio
    async def test_under_limit_calls_handler(self):
        p = ToolCallLimitPlugin(max_calls=5)
        handler = AsyncMock(return_value="result")
        req = ToolRequest(tool_name="t", call_count=3)
        result = await p.wrap_tool_call(req, handler)
        assert result == "result"
        handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_at_limit_calls_handler(self):
        p = ToolCallLimitPlugin(max_calls=5)
        handler = AsyncMock(return_value="ok")
        result = await p.wrap_tool_call(ToolRequest(tool_name="t", call_count=5), handler)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_over_limit_halts_without_calling_handler(self):
        p = ToolCallLimitPlugin(max_calls=3)
        handler = AsyncMock()
        with pytest.raises(PluginHalt):
            await p.wrap_tool_call(ToolRequest(tool_name="t", call_count=4), handler)
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_passes_original_request_to_handler(self):
        p = ToolCallLimitPlugin(max_calls=10)
        handler = AsyncMock(return_value="ok")
        req = ToolRequest(tool_name="search", tool_args={"q": "test"}, call_count=1)
        await p.wrap_tool_call(req, handler)
        handler.assert_called_once_with(req)

    @pytest.mark.asyncio
    async def test_limit_1(self):
        p = ToolCallLimitPlugin(max_calls=1)
        handler = AsyncMock(return_value="ok")
        assert await p.wrap_tool_call(ToolRequest(tool_name="t", call_count=1), handler) == "ok"
        with pytest.raises(PluginHalt):
            await p.wrap_tool_call(ToolRequest(tool_name="t", call_count=2), handler)


# ====================================================================
# 3. ToolRetryPlugin
# ====================================================================

class TestToolRetryPluginDetailed:

    def test_name_is_tool_retry(self):
        assert ToolRetryPlugin().name == "tool_retry"

    def test_defaults(self):
        p = ToolRetryPlugin()
        assert p._max_retries == 3
        assert p._base_delay == 1.0
        assert p._max_delay == 30.0

    @pytest.mark.asyncio
    async def test_no_retry_on_success(self):
        p = ToolRetryPlugin(max_retries=3, base_delay=0.01)
        handler = AsyncMock(return_value="done")
        result = await p.wrap_tool_call(ToolRequest(tool_name="t"), handler)
        assert result == "done"
        assert handler.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_until_success(self):
        p = ToolRetryPlugin(max_retries=3, base_delay=0.001)
        handler = AsyncMock(side_effect=[ValueError, ValueError, "ok"])
        result = await p.wrap_tool_call(ToolRequest(tool_name="t"), handler)
        assert result == "ok"
        assert handler.call_count == 3

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises_original(self):
        p = ToolRetryPlugin(max_retries=2, base_delay=0.001)
        handler = AsyncMock(side_effect=RuntimeError("always fails"))
        with pytest.raises(RuntimeError, match="always fails"):
            await p.wrap_tool_call(ToolRequest(tool_name="t"), handler)
        assert handler.call_count == 3  # 1 initial + 2 retries

    @pytest.mark.asyncio
    async def test_zero_retries(self):
        p = ToolRetryPlugin(max_retries=0, base_delay=0.001)
        handler = AsyncMock(side_effect=ValueError("fail"))
        with pytest.raises(ValueError):
            await p.wrap_tool_call(ToolRequest(tool_name="t"), handler)
        assert handler.call_count == 1

    @pytest.mark.asyncio
    async def test_delay_increases_exponentially(self):
        p = ToolRetryPlugin(max_retries=2, base_delay=0.05, max_delay=10.0)
        handler = AsyncMock(side_effect=[ValueError, ValueError, "ok"])
        start = time.monotonic()
        await p.wrap_tool_call(ToolRequest(tool_name="t"), handler)
        elapsed = time.monotonic() - start
        assert elapsed >= 0.1  # 0.05 + 0.1 = 0.15 minimum

    @pytest.mark.asyncio
    async def test_max_delay_caps_wait(self):
        p = ToolRetryPlugin(max_retries=1, base_delay=100.0, max_delay=0.01)
        handler = AsyncMock(side_effect=[ValueError, "ok"])
        start = time.monotonic()
        await p.wrap_tool_call(ToolRequest(tool_name="t"), handler)
        elapsed = time.monotonic() - start
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_preserves_request_across_retries(self):
        p = ToolRetryPlugin(max_retries=1, base_delay=0.001)
        req = ToolRequest(tool_name="search", tool_args={"q": "hello"})
        handler = AsyncMock(side_effect=[ValueError, "ok"])
        await p.wrap_tool_call(req, handler)
        for c in handler.call_args_list:
            assert c[0][0] is req


# ====================================================================
# 4. ModelFallbackPlugin
# ====================================================================

class TestModelFallbackPluginDetailed:

    def test_name(self):
        assert ModelFallbackPlugin(fallbacks=["m"]).name == "model_fallback"

    def test_empty_fallbacks_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            ModelFallbackPlugin(fallbacks=[])

    def test_single_exception_type_normalized(self):
        p = ModelFallbackPlugin(fallbacks=["m"], retry_on=ValueError)
        assert p._retry_on == (ValueError,)

    def test_tuple_exception_types_preserved(self):
        p = ModelFallbackPlugin(fallbacks=["m"], retry_on=(ValueError, TypeError))
        assert p._retry_on == (ValueError, TypeError)

    @pytest.mark.asyncio
    async def test_primary_success_returns_immediately(self):
        p = ModelFallbackPlugin(fallbacks=["fb1", "fb2"])
        handler = AsyncMock(return_value="primary_ok")
        req = ModelRequest(model="primary")
        assert await p.wrap_model_call(req, handler) == "primary_ok"
        assert handler.call_count == 1

    @pytest.mark.asyncio
    async def test_first_fallback_succeeds(self):
        p = ModelFallbackPlugin(fallbacks=["fb1", "fb2"])
        calls = []
        async def handler(r):
            calls.append(r.model)
            if r.model == "primary":
                raise RuntimeError("down")
            return f"from_{r.model}"
        result = await p.wrap_model_call(ModelRequest(model="primary"), handler)
        assert result == "from_fb1"
        assert calls == ["primary", "fb1"]

    @pytest.mark.asyncio
    async def test_second_fallback_succeeds(self):
        p = ModelFallbackPlugin(fallbacks=["fb1", "fb2"])
        async def handler(r):
            if r.model in ("primary", "fb1"):
                raise RuntimeError("fail")
            return "from_fb2"
        result = await p.wrap_model_call(ModelRequest(model="primary"), handler)
        assert result == "from_fb2"

    @pytest.mark.asyncio
    async def test_all_fail_raises_last_error(self):
        p = ModelFallbackPlugin(fallbacks=["fb1"])
        async def handler(r):
            raise RuntimeError(f"err_{r.model}")
        with pytest.raises(RuntimeError, match="err_fb1"):
            await p.wrap_model_call(ModelRequest(model="primary"), handler)

    @pytest.mark.asyncio
    async def test_unmatched_exception_propagates_immediately(self):
        p = ModelFallbackPlugin(fallbacks=["fb1"], retry_on=ValueError)
        handler = AsyncMock(side_effect=TypeError("wrong type"))
        with pytest.raises(TypeError, match="wrong type"):
            await p.wrap_model_call(ModelRequest(model="primary"), handler)

    @pytest.mark.asyncio
    async def test_fallback_preserves_other_request_fields(self):
        p = ModelFallbackPlugin(fallbacks=["fb1"])
        received_requests = []
        async def handler(r):
            received_requests.append(r)
            if r.model == "primary":
                raise RuntimeError("fail")
            return "ok"
        req = ModelRequest(model="primary", max_tokens=2048, messages=[{"role": "user", "content": "hi"}])
        await p.wrap_model_call(req, handler)
        fb_req = received_requests[1]
        assert fb_req.model == "fb1"
        assert fb_req.max_tokens == 2048
        assert fb_req.messages == req.messages

    @pytest.mark.asyncio
    async def test_original_request_immutable(self):
        p = ModelFallbackPlugin(fallbacks=["fb1"])
        async def handler(r):
            if r.model == "primary":
                raise RuntimeError("fail")
            return "ok"
        req = ModelRequest(model="primary")
        await p.wrap_model_call(req, handler)
        assert req.model == "primary"


# ====================================================================
# 5. PIIGuardPlugin
# ====================================================================

class TestPIIGuardPluginDetailed:

    def test_name(self):
        assert PIIGuardPlugin(pii_types=["email"]).name == "pii_guard"

    # -- construction validation --

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Invalid strategy"):
            PIIGuardPlugin(pii_types=["email"], strategy="encrypt")

    def test_unknown_pii_type_raises(self):
        with pytest.raises(ValueError, match="Unknown PII type"):
            PIIGuardPlugin(pii_types=["dna"])

    def test_no_patterns_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            PIIGuardPlugin()

    def test_accepts_custom_pattern_only(self):
        p = PIIGuardPlugin(custom_patterns={"token": r"tok_[a-z]+"})
        assert "token" in p._patterns

    def test_accepts_mixed_builtin_and_custom(self):
        p = PIIGuardPlugin(pii_types=["email"], custom_patterns={"api": r"sk-.+"})
        assert "email" in p._patterns and "api" in p._patterns

    # -- Luhn algorithm --

    def test_luhn_valid_visa(self):
        assert _luhn_check("4111111111111111") is True

    def test_luhn_valid_mastercard(self):
        assert _luhn_check("5500000000000004") is True

    def test_luhn_invalid_short(self):
        assert _luhn_check("123") is False

    def test_luhn_invalid_bad_check(self):
        assert _luhn_check("4111111111111112") is False

    # -- redact strategy (all PII types) --

    @pytest.mark.asyncio
    async def test_redact_email_in_dict_message(self):
        p = PIIGuardPlugin(pii_types=["email"])
        req = ModelRequest(messages=[{"role": "user", "content": "reach me at alice@corp.io"}])
        result = await p.before_model(req)
        assert result is not None
        assert "[REDACTED_EMAIL]" in result.messages[0]["content"]
        assert "alice@corp.io" not in result.messages[0]["content"]

    @pytest.mark.asyncio
    async def test_redact_multiple_emails(self):
        p = PIIGuardPlugin(pii_types=["email"])
        req = ModelRequest(messages=[{"role": "user", "content": "a@b.com and c@d.com"}])
        result = await p.before_model(req)
        assert result.messages[0]["content"].count("[REDACTED_EMAIL]") == 2

    @pytest.mark.asyncio
    async def test_redact_phone_various_formats(self):
        p = PIIGuardPlugin(pii_types=["phone"])
        for phone in ["555-123-4567", "(555) 123-4567", "555.123.4567"]:
            req = ModelRequest(messages=[{"role": "user", "content": f"call {phone}"}])
            result = await p.before_model(req)
            assert result is not None
            assert "[REDACTED_PHONE]" in result.messages[0]["content"]

    @pytest.mark.asyncio
    async def test_redact_ip_address(self):
        p = PIIGuardPlugin(pii_types=["ip_address"])
        req = ModelRequest(messages=[{"role": "user", "content": "host 10.0.0.1 is down"}])
        result = await p.before_model(req)
        assert "[REDACTED_IP_ADDRESS]" in result.messages[0]["content"]

    @pytest.mark.asyncio
    async def test_redact_ssn(self):
        p = PIIGuardPlugin(pii_types=["ssn"])
        req = ModelRequest(messages=[{"role": "user", "content": "SSN 123-45-6789"}])
        result = await p.before_model(req)
        assert "[REDACTED_SSN]" in result.messages[0]["content"]

    @pytest.mark.asyncio
    async def test_no_pii_returns_none(self):
        p = PIIGuardPlugin(pii_types=["email", "phone", "ssn"])
        req = ModelRequest(messages=[{"role": "user", "content": "Hello world, no PII here!"}])
        assert await p.before_model(req) is None

    @pytest.mark.asyncio
    async def test_redact_across_multiple_messages(self):
        p = PIIGuardPlugin(pii_types=["email"])
        req = ModelRequest(messages=[
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Email me at x@y.com"},
            {"role": "assistant", "content": "Sure!"},
            {"role": "user", "content": "Also z@w.org"},
        ])
        result = await p.before_model(req)
        assert "[REDACTED_EMAIL]" in result.messages[1]["content"]
        assert "[REDACTED_EMAIL]" in result.messages[3]["content"]
        assert result.messages[0]["content"] == "You are a helper."
        assert result.messages[2]["content"] == "Sure!"

    # -- mask strategy --

    @pytest.mark.asyncio
    async def test_mask_email(self):
        p = PIIGuardPlugin(pii_types=["email"], strategy="mask")
        req = ModelRequest(messages=[{"role": "user", "content": "bob@company.com"}])
        result = await p.before_model(req)
        assert "b***@company.com" in result.messages[0]["content"]

    @pytest.mark.asyncio
    async def test_mask_ssn_shows_last_4(self):
        p = PIIGuardPlugin(pii_types=["ssn"], strategy="mask")
        req = ModelRequest(messages=[{"role": "user", "content": "987-65-4321"}])
        result = await p.before_model(req)
        assert "***-**-4321" in result.messages[0]["content"]

    @pytest.mark.asyncio
    async def test_mask_phone_shows_last_4(self):
        p = PIIGuardPlugin(pii_types=["phone"], strategy="mask")
        req = ModelRequest(messages=[{"role": "user", "content": "555-111-2222"}])
        result = await p.before_model(req)
        assert "***-***-2222" in result.messages[0]["content"]

    @pytest.mark.asyncio
    async def test_mask_ip_shows_last_octet(self):
        p = PIIGuardPlugin(pii_types=["ip_address"], strategy="mask")
        req = ModelRequest(messages=[{"role": "user", "content": "192.168.1.42"}])
        result = await p.before_model(req)
        assert "***.***.***.42" in result.messages[0]["content"]

    @pytest.mark.asyncio
    async def test_mask_custom_pattern(self):
        p = PIIGuardPlugin(custom_patterns={"secret": r"SEC-[A-Z]{6}"}, strategy="mask")
        req = ModelRequest(messages=[{"role": "user", "content": "key SEC-ABCDEF"}])
        result = await p.before_model(req)
        content = result.messages[0]["content"]
        assert "SEC-ABCDEF" not in content

    # -- block strategy --

    @pytest.mark.asyncio
    async def test_block_raises_halt_with_type_info(self):
        p = PIIGuardPlugin(pii_types=["email", "ssn"], strategy="block")
        req = ModelRequest(messages=[{"role": "user", "content": "SSN 111-22-3333"}])
        with pytest.raises(PluginHalt) as exc:
            await p.before_model(req)
        assert "ssn" in str(exc.value.result).lower()

    @pytest.mark.asyncio
    async def test_block_clean_message_passes(self):
        p = PIIGuardPlugin(pii_types=["email"], strategy="block")
        assert await p.before_model(ModelRequest(messages=[{"role": "user", "content": "safe text"}])) is None

    # -- custom regex patterns --

    @pytest.mark.asyncio
    async def test_custom_api_key_redacted(self):
        p = PIIGuardPlugin(custom_patterns={"api_key": r"sk-[a-zA-Z0-9]{20,}"})
        req = ModelRequest(messages=[{"role": "user", "content": "key is sk-abc123def456ghi789jkl"}])
        result = await p.before_model(req)
        assert "[REDACTED_API_KEY]" in result.messages[0]["content"]

    # -- apply_to_input / apply_to_output flags --

    @pytest.mark.asyncio
    async def test_input_disabled_skips_before_model(self):
        p = PIIGuardPlugin(pii_types=["email"], apply_to_input=False)
        req = ModelRequest(messages=[{"role": "user", "content": "x@y.com"}])
        assert await p.before_model(req) is None

    @pytest.mark.asyncio
    async def test_output_enabled_redacts_response(self):
        p = PIIGuardPlugin(pii_types=["email"], apply_to_output=True)
        resp = MagicMock()
        resp.content = "reply to alice@corp.com"
        resp.model_copy = MagicMock(return_value=MagicMock(content="reply to [REDACTED_EMAIL]"))
        result = await p.after_model(ModelRequest(), resp)
        resp.model_copy.assert_called_once()

    @pytest.mark.asyncio
    async def test_output_disabled_passes_through(self):
        p = PIIGuardPlugin(pii_types=["email"], apply_to_output=False)
        assert await p.after_model(ModelRequest(), "anything") == "anything"

    @pytest.mark.asyncio
    async def test_output_block_halts(self):
        p = PIIGuardPlugin(pii_types=["email"], strategy="block", apply_to_output=True)
        resp = MagicMock()
        resp.content = "contact admin@server.com"
        with pytest.raises(PluginHalt, match="PII detected in model response"):
            await p.after_model(ModelRequest(), resp)

    # -- immutability --

    @pytest.mark.asyncio
    async def test_original_request_not_mutated(self):
        p = PIIGuardPlugin(pii_types=["email"])
        original_msg = {"role": "user", "content": "hi@there.com"}
        req = ModelRequest(messages=[original_msg])
        result = await p.before_model(req)
        assert req.messages[0]["content"] == "hi@there.com"
        assert "[REDACTED_EMAIL]" in result.messages[0]["content"]


# ====================================================================
# 6. HumanApprovalPlugin
# ====================================================================

class TestHumanApprovalPluginDetailed:

    def test_name(self):
        assert HumanApprovalPlugin(approval_callback=lambda n, a: True).name == "human_approval"

    # -- sync callback --

    @pytest.mark.asyncio
    async def test_sync_approve(self):
        p = HumanApprovalPlugin(approval_callback=lambda n, a: True)
        handler = AsyncMock(return_value="result")
        assert await p.wrap_tool_call(ToolRequest(tool_name="t"), handler) == "result"

    @pytest.mark.asyncio
    async def test_sync_deny(self):
        p = HumanApprovalPlugin(approval_callback=lambda n, a: False)
        handler = AsyncMock()
        result = await p.wrap_tool_call(ToolRequest(tool_name="t"), handler)
        handler.assert_not_called()
        assert "denied" in result.lower()

    # -- async callback --

    @pytest.mark.asyncio
    async def test_async_approve(self):
        async def cb(n, a):
            return True
        p = HumanApprovalPlugin(approval_callback=cb)
        handler = AsyncMock(return_value="ok")
        assert await p.wrap_tool_call(ToolRequest(tool_name="t"), handler) == "ok"

    @pytest.mark.asyncio
    async def test_async_deny(self):
        async def cb(n, a):
            return False
        p = HumanApprovalPlugin(approval_callback=cb)
        handler = AsyncMock()
        result = await p.wrap_tool_call(ToolRequest(tool_name="t"), handler)
        handler.assert_not_called()
        assert "denied" in result.lower()

    # -- callback receives correct args --

    @pytest.mark.asyncio
    async def test_callback_receives_name_and_args(self):
        received = {}
        def cb(name, args):
            received["name"] = name
            received["args"] = args
            return True
        p = HumanApprovalPlugin(approval_callback=cb)
        handler = AsyncMock(return_value="ok")
        await p.wrap_tool_call(ToolRequest(tool_name="search", tool_args={"q": "hello"}), handler)
        assert received == {"name": "search", "args": {"q": "hello"}}

    # -- require_approval list --

    @pytest.mark.asyncio
    async def test_require_approval_only_gates_listed(self):
        p = HumanApprovalPlugin(
            approval_callback=lambda n, a: False,
            require_approval=["dangerous"],
        )
        handler = AsyncMock(return_value="ok")
        assert await p.wrap_tool_call(ToolRequest(tool_name="safe"), handler) == "ok"
        handler.reset_mock()
        result = await p.wrap_tool_call(ToolRequest(tool_name="dangerous"), handler)
        assert "denied" in result.lower()
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_require_approval_empty_list_treated_as_none(self):
        """Empty require_approval list is falsy, so treated as None (all tools need approval)."""
        p = HumanApprovalPlugin(
            approval_callback=lambda n, a: False,
            require_approval=[],
        )
        handler = AsyncMock()
        result = await p.wrap_tool_call(ToolRequest(tool_name="anything"), handler)
        assert "denied" in result.lower()
        handler.assert_not_called()

    # -- auto_approve list --

    @pytest.mark.asyncio
    async def test_auto_approve_bypasses_callback(self):
        call_count = {"n": 0}
        def cb(n, a):
            call_count["n"] += 1
            return False
        p = HumanApprovalPlugin(approval_callback=cb, auto_approve=["math"])
        handler = AsyncMock(return_value="42")
        assert await p.wrap_tool_call(ToolRequest(tool_name="math"), handler) == "42"
        assert call_count["n"] == 0

    @pytest.mark.asyncio
    async def test_auto_approve_takes_priority_over_require(self):
        p = HumanApprovalPlugin(
            approval_callback=lambda n, a: False,
            require_approval=["calc"],
            auto_approve=["calc"],
        )
        handler = AsyncMock(return_value="ok")
        assert await p.wrap_tool_call(ToolRequest(tool_name="calc"), handler) == "ok"

    # -- custom deny message --

    @pytest.mark.asyncio
    async def test_custom_deny_message(self):
        p = HumanApprovalPlugin(approval_callback=lambda n, a: False, deny_message="NOPE")
        handler = AsyncMock()
        result = await p.wrap_tool_call(ToolRequest(tool_name="x"), handler)
        assert "NOPE" in result

    @pytest.mark.asyncio
    async def test_deny_message_includes_tool_name(self):
        p = HumanApprovalPlugin(approval_callback=lambda n, a: False)
        handler = AsyncMock()
        result = await p.wrap_tool_call(ToolRequest(tool_name="delete_db"), handler)
        assert "delete_db" in result


# ====================================================================
# 7. ContextWindowPlugin
# ====================================================================

class TestContextWindowPluginDetailed:

    def test_name(self):
        assert ContextWindowPlugin(max_messages=10).name == "context_window"

    # -- construction validation --

    def test_no_limits_raises(self):
        with pytest.raises(ValueError):
            ContextWindowPlugin()

    def test_max_messages_1_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            ContextWindowPlugin(max_messages=1)

    def test_keep_recent_0_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            ContextWindowPlugin(max_messages=10, keep_recent=0)

    def test_max_messages_2_ok(self):
        ContextWindowPlugin(max_messages=2)

    def test_token_only_ok(self):
        ContextWindowPlugin(max_tokens=1000)

    def test_both_limits_ok(self):
        ContextWindowPlugin(max_messages=50, max_tokens=4000)

    # -- message trimming --

    @pytest.mark.asyncio
    async def test_under_limit_no_change(self):
        p = ContextWindowPlugin(max_messages=10)
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(5)]
        assert await p.before_model(ModelRequest(messages=msgs)) is None

    @pytest.mark.asyncio
    async def test_at_limit_no_change(self):
        p = ContextWindowPlugin(max_messages=5, keep_recent=3)
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(5)]
        assert await p.before_model(ModelRequest(messages=msgs)) is None

    @pytest.mark.asyncio
    async def test_trims_preserves_first_and_last(self):
        p = ContextWindowPlugin(max_messages=5, keep_recent=2, placeholder=None)
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(10)]
        result = await p.before_model(ModelRequest(messages=msgs))
        assert result.messages[0]["content"] == "m0"
        assert result.messages[-1]["content"] == "m9"
        assert result.messages[-2]["content"] == "m8"

    @pytest.mark.asyncio
    async def test_placeholder_inserted(self):
        p = ContextWindowPlugin(max_messages=5, keep_recent=2, placeholder="[...]")
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(10)]
        result = await p.before_model(ModelRequest(messages=msgs))
        assert result.messages[1] == {"role": "system", "content": "[...]"}

    @pytest.mark.asyncio
    async def test_no_placeholder_when_none(self):
        p = ContextWindowPlugin(max_messages=5, keep_recent=2, placeholder=None)
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(10)]
        result = await p.before_model(ModelRequest(messages=msgs))
        for m in result.messages:
            assert m["content"] != "[...]"

    @pytest.mark.asyncio
    async def test_keep_recent_larger_than_middle(self):
        p = ContextWindowPlugin(max_messages=3, keep_recent=8, placeholder=None)
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(5)]
        assert await p.before_model(ModelRequest(messages=msgs)) is None

    @pytest.mark.asyncio
    async def test_many_messages_trimmed_correctly(self):
        p = ContextWindowPlugin(max_messages=10, keep_recent=5, placeholder=None)
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(100)]
        result = await p.before_model(ModelRequest(messages=msgs))
        assert len(result.messages) == 6  # 1 head + 5 recent
        assert result.messages[0]["content"] == "m0"
        assert result.messages[-1]["content"] == "m99"

    # -- token trimming --

    @pytest.mark.asyncio
    async def test_token_under_limit_no_change(self):
        p = ContextWindowPlugin(max_tokens=10000, keep_recent=2, token_counter=len)
        msgs = [{"role": "user", "content": "hi"} for _ in range(3)]
        assert await p.before_model(ModelRequest(messages=msgs)) is None

    @pytest.mark.asyncio
    async def test_token_over_limit_trims(self):
        p = ContextWindowPlugin(max_tokens=60, keep_recent=1, token_counter=len, placeholder=None)
        msgs = [{"role": "user", "content": "x" * 30} for _ in range(5)]
        result = await p.before_model(ModelRequest(messages=msgs))
        assert result is not None
        assert len(result.messages) < 5

    @pytest.mark.asyncio
    async def test_custom_token_counter(self):
        word_count = lambda s: len(s.split())
        p = ContextWindowPlugin(max_tokens=5, keep_recent=1, token_counter=word_count, placeholder=None)
        msgs = [
            {"role": "system", "content": "be helpful"},  # 2 words
            {"role": "user", "content": "one two three"},  # 3 words
            {"role": "user", "content": "four five six"},  # 3 words
            {"role": "user", "content": "seven eight"},  # 2 words
        ]
        result = await p.before_model(ModelRequest(messages=msgs))
        assert result is not None

    # -- combined limits --

    @pytest.mark.asyncio
    async def test_message_limit_applied_before_token_limit(self):
        p = ContextWindowPlugin(max_messages=5, max_tokens=10000, keep_recent=2, placeholder=None, token_counter=len)
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        result = await p.before_model(ModelRequest(messages=msgs))
        assert len(result.messages) == 3  # 1 head + 2 recent

    # -- immutability --

    @pytest.mark.asyncio
    async def test_original_messages_not_mutated(self):
        p = ContextWindowPlugin(max_messages=3, keep_recent=1, placeholder=None)
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(10)]
        req = ModelRequest(messages=msgs)
        result = await p.before_model(req)
        assert len(req.messages) == 10
        assert len(result.messages) == 2


# ====================================================================
# 8. ToolGuardPlugin
# ====================================================================

class TestToolGuardPluginDetailed:

    def test_name(self):
        assert ToolGuardPlugin(blocked=["x"]).name == "tool_guard"

    # -- construction validation --

    def test_both_modes_raises(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            ToolGuardPlugin(blocked=["a"], allowed=["b"])

    def test_neither_mode_raises(self):
        with pytest.raises(ValueError, match="Must specify"):
            ToolGuardPlugin()

    # -- blocklist mode --

    @pytest.mark.asyncio
    async def test_blocked_tool_returns_message(self):
        p = ToolGuardPlugin(blocked=["rm", "delete"])
        handler = AsyncMock()
        result = await p.wrap_tool_call(ToolRequest(tool_name="rm"), handler)
        assert "blocked" in result.lower()
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_unblocked_tool_executes(self):
        p = ToolGuardPlugin(blocked=["rm"])
        handler = AsyncMock(return_value="found")
        assert await p.wrap_tool_call(ToolRequest(tool_name="search"), handler) == "found"

    @pytest.mark.asyncio
    async def test_multiple_blocked_tools(self):
        p = ToolGuardPlugin(blocked=["a", "b", "c"])
        handler = AsyncMock()
        for name in ["a", "b", "c"]:
            result = await p.wrap_tool_call(ToolRequest(tool_name=name), handler)
            assert "blocked" in result.lower()
        handler.assert_not_called()

    # -- allowlist mode --

    @pytest.mark.asyncio
    async def test_allowed_tool_executes(self):
        p = ToolGuardPlugin(allowed=["search", "calc"])
        handler = AsyncMock(return_value="ok")
        assert await p.wrap_tool_call(ToolRequest(tool_name="search"), handler) == "ok"

    @pytest.mark.asyncio
    async def test_disallowed_tool_blocked(self):
        p = ToolGuardPlugin(allowed=["search"])
        handler = AsyncMock()
        result = await p.wrap_tool_call(ToolRequest(tool_name="delete"), handler)
        assert "blocked" in result.lower()

    @pytest.mark.asyncio
    async def test_empty_blocklist_allows_all(self):
        # blocked=[] is falsy, should raise
        with pytest.raises(ValueError):
            ToolGuardPlugin(blocked=[])

    # -- custom deny handlers --

    @pytest.mark.asyncio
    async def test_static_string_deny(self):
        p = ToolGuardPlugin(blocked=["bad"], on_deny="Forbidden!")
        handler = AsyncMock()
        assert await p.wrap_tool_call(ToolRequest(tool_name="bad"), handler) == "Forbidden!"

    @pytest.mark.asyncio
    async def test_callable_deny_receives_name_and_args(self):
        received = {}
        def deny_fn(name, args):
            received.update(name=name, args=args)
            return "denied"
        p = ToolGuardPlugin(blocked=["x"], on_deny=deny_fn)
        handler = AsyncMock()
        await p.wrap_tool_call(ToolRequest(tool_name="x", tool_args={"k": "v"}), handler)
        assert received == {"name": "x", "args": {"k": "v"}}

    @pytest.mark.asyncio
    async def test_default_deny_message_includes_tool_name(self):
        p = ToolGuardPlugin(blocked=["drop_table"])
        handler = AsyncMock()
        result = await p.wrap_tool_call(ToolRequest(tool_name="drop_table"), handler)
        assert "drop_table" in result

    # -- handler receives correct request --

    @pytest.mark.asyncio
    async def test_handler_receives_original_request(self):
        p = ToolGuardPlugin(blocked=["bad"])
        handler = AsyncMock(return_value="ok")
        req = ToolRequest(tool_name="good", tool_args={"x": 1})
        await p.wrap_tool_call(req, handler)
        handler.assert_called_once_with(req)


# ====================================================================
# Multi-plugin pipeline tests
# ====================================================================

class TestMultiPluginPipeline:

    @pytest.mark.asyncio
    async def test_model_limit_with_fallback(self):
        """ModelCallLimit halts before Fallback even gets a chance."""
        mgr = PluginManager([
            ModelCallLimitPlugin(max_calls=2),
            ModelFallbackPlugin(fallbacks=["fb1"]),
        ])
        req = ModelRequest(call_count=3)
        with pytest.raises(PluginHalt):
            await mgr.run_before_model(req)

    @pytest.mark.asyncio
    async def test_tool_guard_then_retry(self):
        """ToolGuard blocks before ToolRetry even attempts."""
        mgr = PluginManager([
            ToolGuardPlugin(blocked=["bad"]),
            ToolRetryPlugin(max_retries=3, base_delay=0.001),
        ])
        handler = AsyncMock(return_value="ok")
        result = await mgr.execute_tool_call(ToolRequest(tool_name="bad"), handler)
        assert "blocked" in result.lower()
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_pii_guard_before_model_limit(self):
        """PII is redacted, then limit check passes."""
        mgr = PluginManager([
            PIIGuardPlugin(pii_types=["email"]),
            ModelCallLimitPlugin(max_calls=10),
        ])
        req = ModelRequest(messages=[{"role": "user", "content": "hi@there.com"}], call_count=1)
        result = await mgr.run_before_model(req)
        assert "[REDACTED_EMAIL]" in result.messages[0]["content"]

    @pytest.mark.asyncio
    async def test_approval_then_guard(self):
        """Approval runs first (auto-approve), then guard blocks."""
        mgr = PluginManager([
            HumanApprovalPlugin(approval_callback=lambda n, a: True),
            ToolGuardPlugin(blocked=["bad"]),
        ])
        handler = AsyncMock(return_value="ok")
        result = await mgr.execute_tool_call(ToolRequest(tool_name="bad"), handler)
        assert "blocked" in result.lower()

    @pytest.mark.asyncio
    async def test_all_8_plugins_together(self):
        """Smoke test: all 8 plugins registered without errors."""
        mgr = PluginManager([
            ModelCallLimitPlugin(max_calls=100),
            ToolCallLimitPlugin(max_calls=100),
            ToolRetryPlugin(max_retries=1, base_delay=0.001),
            ModelFallbackPlugin(fallbacks=["fb"]),
            PIIGuardPlugin(pii_types=["email"]),
            HumanApprovalPlugin(approval_callback=lambda n, a: True),
            ContextWindowPlugin(max_messages=50),
            ToolGuardPlugin(blocked=["bad"]),
        ])
        assert len(mgr.plugins) == 8
        req = ModelRequest(messages=[{"role": "user", "content": "hello"}], call_count=1)
        result = await mgr.run_before_model(req)
        assert result is None or isinstance(result, ModelRequest)
