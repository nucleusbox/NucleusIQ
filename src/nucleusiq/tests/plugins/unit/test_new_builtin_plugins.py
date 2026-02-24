"""Unit tests for the 5 new built-in plugins."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from nucleusiq.plugins.base import ModelRequest, ToolRequest
from nucleusiq.plugins.builtin.context_window import ContextWindowPlugin
from nucleusiq.plugins.builtin.human_approval import HumanApprovalPlugin
from nucleusiq.plugins.builtin.model_fallback import ModelFallbackPlugin
from nucleusiq.plugins.builtin.pii_guard import PIIGuardPlugin, _luhn_check
from nucleusiq.plugins.builtin.tool_guard import ToolGuardPlugin
from nucleusiq.plugins.errors import PluginHalt

# ==================================================================== #
# ModelFallbackPlugin                                                    #
# ==================================================================== #


class TestModelFallbackPlugin:
    def test_name(self):
        p = ModelFallbackPlugin(fallbacks=["gpt-4o-mini"])
        assert p.name == "model_fallback"

    def test_no_fallbacks_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            ModelFallbackPlugin(fallbacks=[])

    @pytest.mark.asyncio
    async def test_primary_succeeds_no_fallback(self):
        p = ModelFallbackPlugin(fallbacks=["gpt-4o-mini"])
        req = ModelRequest(model="gpt-4o", agent_name="a")
        handler = AsyncMock(return_value="primary_result")

        result = await p.wrap_model_call(req, handler)
        assert result == "primary_result"
        handler.assert_called_once_with(req)

    @pytest.mark.asyncio
    async def test_primary_fails_first_fallback_succeeds(self):
        p = ModelFallbackPlugin(fallbacks=["gpt-4o-mini", "gpt-3.5-turbo"])
        req = ModelRequest(model="gpt-4o", agent_name="a")

        call_models = []

        async def handler(r: ModelRequest):
            call_models.append(r.model)
            if r.model == "gpt-4o":
                raise RuntimeError("primary down")
            return f"result_from_{r.model}"

        result = await p.wrap_model_call(req, handler)
        assert result == "result_from_gpt-4o-mini"
        assert call_models == ["gpt-4o", "gpt-4o-mini"]

    @pytest.mark.asyncio
    async def test_all_models_fail_raises_last(self):
        p = ModelFallbackPlugin(fallbacks=["fb1", "fb2"])
        req = ModelRequest(model="primary", agent_name="a")

        async def handler(r: ModelRequest):
            raise RuntimeError(f"fail_{r.model}")

        with pytest.raises(RuntimeError, match="fail_fb2"):
            await p.wrap_model_call(req, handler)

    @pytest.mark.asyncio
    async def test_retry_on_specific_exception(self):
        p = ModelFallbackPlugin(
            fallbacks=["fb1"],
            retry_on=ValueError,
        )
        req = ModelRequest(model="primary", agent_name="a")

        async def handler(r: ModelRequest):
            if r.model == "primary":
                raise ValueError("bad value")
            return "ok"

        result = await p.wrap_model_call(req, handler)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_non_matching_exception_not_caught(self):
        p = ModelFallbackPlugin(
            fallbacks=["fb1"],
            retry_on=ValueError,
        )
        req = ModelRequest(model="primary", agent_name="a")
        handler = AsyncMock(side_effect=RuntimeError("unexpected"))

        with pytest.raises(RuntimeError, match="unexpected"):
            await p.wrap_model_call(req, handler)

    @pytest.mark.asyncio
    async def test_second_fallback_used_when_first_fails(self):
        p = ModelFallbackPlugin(fallbacks=["fb1", "fb2", "fb3"])
        req = ModelRequest(model="primary", agent_name="a")

        async def handler(r: ModelRequest):
            if r.model in ("primary", "fb1", "fb2"):
                raise RuntimeError(f"fail_{r.model}")
            return "from_fb3"

        result = await p.wrap_model_call(req, handler)
        assert result == "from_fb3"


# ==================================================================== #
# PIIGuardPlugin                                                         #
# ==================================================================== #


class TestPIIGuardPlugin:
    def test_name(self):
        p = PIIGuardPlugin(pii_types=["email"])
        assert p.name == "pii_guard"

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="Invalid strategy"):
            PIIGuardPlugin(pii_types=["email"], strategy="encrypt")

    def test_unknown_pii_type(self):
        with pytest.raises(ValueError, match="Unknown PII type"):
            PIIGuardPlugin(pii_types=["fingerprint"])

    def test_no_patterns_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            PIIGuardPlugin()

    def test_luhn_valid(self):
        assert _luhn_check("4111111111111111") is True

    def test_luhn_invalid(self):
        assert _luhn_check("1234567890") is False

    # -- redact strategy --

    @pytest.mark.asyncio
    async def test_redact_email(self):
        p = PIIGuardPlugin(pii_types=["email"], strategy="redact")
        req = ModelRequest(
            messages=[{"role": "user", "content": "My email is john@example.com"}],
            agent_name="a",
        )
        result = await p.before_model(req)
        assert result is not None
        assert "[REDACTED_EMAIL]" in result.messages[0]["content"]
        assert "john@example.com" not in result.messages[0]["content"]

    @pytest.mark.asyncio
    async def test_redact_phone(self):
        p = PIIGuardPlugin(pii_types=["phone"], strategy="redact")
        req = ModelRequest(
            messages=[{"role": "user", "content": "Call me at 555-123-4567"}],
            agent_name="a",
        )
        result = await p.before_model(req)
        assert result is not None
        assert "[REDACTED_PHONE]" in result.messages[0]["content"]

    @pytest.mark.asyncio
    async def test_redact_ssn(self):
        p = PIIGuardPlugin(pii_types=["ssn"], strategy="redact")
        req = ModelRequest(
            messages=[{"role": "user", "content": "SSN: 123-45-6789"}],
            agent_name="a",
        )
        result = await p.before_model(req)
        assert result is not None
        assert "[REDACTED_SSN]" in result.messages[0]["content"]

    @pytest.mark.asyncio
    async def test_redact_ip(self):
        p = PIIGuardPlugin(pii_types=["ip_address"], strategy="redact")
        req = ModelRequest(
            messages=[{"role": "user", "content": "Server at 192.168.1.100"}],
            agent_name="a",
        )
        result = await p.before_model(req)
        assert result is not None
        assert "[REDACTED_IP_ADDRESS]" in result.messages[0]["content"]

    @pytest.mark.asyncio
    async def test_no_pii_returns_none(self):
        p = PIIGuardPlugin(pii_types=["email"], strategy="redact")
        req = ModelRequest(
            messages=[{"role": "user", "content": "Hello world"}],
            agent_name="a",
        )
        result = await p.before_model(req)
        assert result is None

    # -- mask strategy --

    @pytest.mark.asyncio
    async def test_mask_email(self):
        p = PIIGuardPlugin(pii_types=["email"], strategy="mask")
        req = ModelRequest(
            messages=[{"role": "user", "content": "Email: john@example.com"}],
            agent_name="a",
        )
        result = await p.before_model(req)
        assert result is not None
        assert "j***@example.com" in result.messages[0]["content"]

    @pytest.mark.asyncio
    async def test_mask_ssn(self):
        p = PIIGuardPlugin(pii_types=["ssn"], strategy="mask")
        req = ModelRequest(
            messages=[{"role": "user", "content": "SSN: 123-45-6789"}],
            agent_name="a",
        )
        result = await p.before_model(req)
        assert result is not None
        assert "***-**-6789" in result.messages[0]["content"]

    # -- block strategy --

    @pytest.mark.asyncio
    async def test_block_halts_on_pii(self):
        p = PIIGuardPlugin(pii_types=["email"], strategy="block")
        req = ModelRequest(
            messages=[{"role": "user", "content": "Email: test@test.com"}],
            agent_name="a",
        )
        with pytest.raises(PluginHalt) as exc_info:
            await p.before_model(req)
        assert "PII detected" in str(exc_info.value.result)

    @pytest.mark.asyncio
    async def test_block_no_pii_passes(self):
        p = PIIGuardPlugin(pii_types=["email"], strategy="block")
        req = ModelRequest(
            messages=[{"role": "user", "content": "No sensitive data here."}],
            agent_name="a",
        )
        result = await p.before_model(req)
        assert result is None

    # -- custom patterns --

    @pytest.mark.asyncio
    async def test_custom_pattern(self):
        p = PIIGuardPlugin(
            custom_patterns={"api_key": r"sk-[a-zA-Z0-9]{10,}"},
            strategy="redact",
        )
        req = ModelRequest(
            messages=[{"role": "user", "content": "Key: sk-abcdef1234567890"}],
            agent_name="a",
        )
        result = await p.before_model(req)
        assert result is not None
        assert "[REDACTED_API_KEY]" in result.messages[0]["content"]

    # -- apply_to_output --

    @pytest.mark.asyncio
    async def test_after_model_redacts_output(self):
        p = PIIGuardPlugin(pii_types=["email"], strategy="redact", apply_to_output=True)
        req = ModelRequest(agent_name="a")
        response = MagicMock()
        response.content = "Contact alice@company.com for details"
        response.model_copy = MagicMock(
            return_value=MagicMock(content="Contact [REDACTED_EMAIL] for details")
        )

        result = await p.after_model(req, response)
        response.model_copy.assert_called_once()

    @pytest.mark.asyncio
    async def test_after_model_skips_when_disabled(self):
        p = PIIGuardPlugin(
            pii_types=["email"], strategy="redact", apply_to_output=False
        )
        req = ModelRequest(agent_name="a")
        response = "plain string"
        result = await p.after_model(req, response)
        assert result == "plain string"

    # -- apply_to_input disabled --

    @pytest.mark.asyncio
    async def test_input_disabled(self):
        p = PIIGuardPlugin(pii_types=["email"], strategy="redact", apply_to_input=False)
        req = ModelRequest(
            messages=[{"role": "user", "content": "john@test.com"}],
            agent_name="a",
        )
        result = await p.before_model(req)
        assert result is None


# ==================================================================== #
# HumanApprovalPlugin                                                   #
# ==================================================================== #


class TestHumanApprovalPlugin:
    def test_name(self):
        p = HumanApprovalPlugin(approval_callback=lambda n, a: True)
        assert p.name == "human_approval"

    @pytest.mark.asyncio
    async def test_approved_executes_tool(self):
        p = HumanApprovalPlugin(approval_callback=lambda n, a: True)
        req = ToolRequest(tool_name="search", tool_args={"q": "test"}, agent_name="a")
        handler = AsyncMock(return_value="search_result")

        result = await p.wrap_tool_call(req, handler)
        assert result == "search_result"
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_denied_returns_message(self):
        p = HumanApprovalPlugin(approval_callback=lambda n, a: False)
        req = ToolRequest(tool_name="delete", tool_args={}, agent_name="a")
        handler = AsyncMock()

        result = await p.wrap_tool_call(req, handler)
        assert "denied" in result.lower()
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_callback(self):
        async def async_approve(name, args):
            return name == "safe_tool"

        p = HumanApprovalPlugin(approval_callback=async_approve)

        safe_req = ToolRequest(tool_name="safe_tool", tool_args={}, agent_name="a")
        handler = AsyncMock(return_value="ok")
        result = await p.wrap_tool_call(safe_req, handler)
        assert result == "ok"

        danger_req = ToolRequest(tool_name="danger_tool", tool_args={}, agent_name="a")
        handler2 = AsyncMock()
        result2 = await p.wrap_tool_call(danger_req, handler2)
        assert "denied" in result2.lower()

    @pytest.mark.asyncio
    async def test_require_approval_filter(self):
        p = HumanApprovalPlugin(
            approval_callback=lambda n, a: False,
            require_approval=["dangerous"],
        )

        safe_req = ToolRequest(tool_name="search", tool_args={}, agent_name="a")
        handler = AsyncMock(return_value="ok")
        result = await p.wrap_tool_call(safe_req, handler)
        assert result == "ok"

        danger_req = ToolRequest(tool_name="dangerous", tool_args={}, agent_name="a")
        handler2 = AsyncMock()
        result2 = await p.wrap_tool_call(danger_req, handler2)
        assert "denied" in result2.lower()

    @pytest.mark.asyncio
    async def test_auto_approve_bypasses(self):
        p = HumanApprovalPlugin(
            approval_callback=lambda n, a: False,
            auto_approve=["calculator"],
        )

        req = ToolRequest(tool_name="calculator", tool_args={"x": 1}, agent_name="a")
        handler = AsyncMock(return_value="42")
        result = await p.wrap_tool_call(req, handler)
        assert result == "42"

    @pytest.mark.asyncio
    async def test_custom_deny_message(self):
        p = HumanApprovalPlugin(
            approval_callback=lambda n, a: False,
            deny_message="Access forbidden.",
        )
        req = ToolRequest(tool_name="rm", tool_args={}, agent_name="a")
        handler = AsyncMock()
        result = await p.wrap_tool_call(req, handler)
        assert "Access forbidden" in result


# ==================================================================== #
# ContextWindowPlugin                                                    #
# ==================================================================== #


class TestContextWindowPlugin:
    def test_name(self):
        p = ContextWindowPlugin(max_messages=10)
        assert p.name == "context_window"

    def test_no_limits_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            ContextWindowPlugin()

    def test_small_max_messages_raises(self):
        with pytest.raises(ValueError, match="max_messages must be at least 2"):
            ContextWindowPlugin(max_messages=1)

    def test_zero_keep_recent_raises(self):
        with pytest.raises(ValueError, match="keep_recent must be at least 1"):
            ContextWindowPlugin(max_messages=10, keep_recent=0)

    @pytest.mark.asyncio
    async def test_under_limit_returns_none(self):
        p = ContextWindowPlugin(max_messages=10, keep_recent=3)
        msgs = [{"role": "user", "content": f"msg{i}"} for i in range(5)]
        req = ModelRequest(messages=msgs, agent_name="a")
        result = await p.before_model(req)
        assert result is None

    @pytest.mark.asyncio
    async def test_trims_by_message_count(self):
        p = ContextWindowPlugin(max_messages=5, keep_recent=3, placeholder=None)
        msgs = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
        req = ModelRequest(messages=msgs, agent_name="a")
        result = await p.before_model(req)

        assert result is not None
        assert len(result.messages) == 4  # 1 head + 3 recent
        assert result.messages[0]["content"] == "msg0"
        assert result.messages[-1]["content"] == "msg9"

    @pytest.mark.asyncio
    async def test_trims_with_placeholder(self):
        p = ContextWindowPlugin(
            max_messages=5,
            keep_recent=2,
            placeholder="[trimmed]",
        )
        msgs = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
        req = ModelRequest(messages=msgs, agent_name="a")
        result = await p.before_model(req)

        assert result is not None
        assert result.messages[0]["content"] == "msg0"
        assert result.messages[1]["content"] == "[trimmed]"
        assert result.messages[-1]["content"] == "msg9"

    @pytest.mark.asyncio
    async def test_trims_by_token_count(self):
        p = ContextWindowPlugin(
            max_tokens=50,
            keep_recent=2,
            token_counter=lambda s: len(s),
            placeholder=None,
        )
        msgs = [{"role": "user", "content": "x" * 20} for _ in range(10)]
        req = ModelRequest(messages=msgs, agent_name="a")
        result = await p.before_model(req)

        assert result is not None
        assert len(result.messages) < 10

    @pytest.mark.asyncio
    async def test_token_under_limit_returns_none(self):
        p = ContextWindowPlugin(
            max_tokens=10000,
            keep_recent=2,
            token_counter=lambda s: len(s),
        )
        msgs = [{"role": "user", "content": "hi"} for _ in range(3)]
        req = ModelRequest(messages=msgs, agent_name="a")
        result = await p.before_model(req)
        assert result is None

    @pytest.mark.asyncio
    async def test_combined_message_and_token_limit(self):
        p = ContextWindowPlugin(
            max_messages=20,
            max_tokens=100,
            keep_recent=2,
            token_counter=lambda s: len(s),
            placeholder=None,
        )
        msgs = [{"role": "user", "content": "x" * 30} for _ in range(10)]
        req = ModelRequest(messages=msgs, agent_name="a")
        result = await p.before_model(req)
        assert result is not None
        assert len(result.messages) < 10


# ==================================================================== #
# ToolGuardPlugin                                                        #
# ==================================================================== #


class TestToolGuardPlugin:
    def test_name(self):
        p = ToolGuardPlugin(blocked=["rm"])
        assert p.name == "tool_guard"

    def test_both_raises(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            ToolGuardPlugin(blocked=["x"], allowed=["y"])

    def test_neither_raises(self):
        with pytest.raises(ValueError, match="Must specify"):
            ToolGuardPlugin()

    @pytest.mark.asyncio
    async def test_blocklist_allows_safe_tools(self):
        p = ToolGuardPlugin(blocked=["delete", "drop"])
        req = ToolRequest(tool_name="search", tool_args={}, agent_name="a")
        handler = AsyncMock(return_value="found it")

        result = await p.wrap_tool_call(req, handler)
        assert result == "found it"
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_blocklist_blocks_dangerous_tools(self):
        p = ToolGuardPlugin(blocked=["delete", "drop"])
        req = ToolRequest(tool_name="delete", tool_args={}, agent_name="a")
        handler = AsyncMock()

        result = await p.wrap_tool_call(req, handler)
        assert "blocked" in result.lower()
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_allowlist_permits_listed(self):
        p = ToolGuardPlugin(allowed=["search", "calc"])
        req = ToolRequest(tool_name="search", tool_args={}, agent_name="a")
        handler = AsyncMock(return_value="ok")

        result = await p.wrap_tool_call(req, handler)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_allowlist_denies_unlisted(self):
        p = ToolGuardPlugin(allowed=["search", "calc"])
        req = ToolRequest(tool_name="delete_all", tool_args={}, agent_name="a")
        handler = AsyncMock()

        result = await p.wrap_tool_call(req, handler)
        assert "blocked" in result.lower()
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_custom_deny_message_string(self):
        p = ToolGuardPlugin(blocked=["bad"], on_deny="Nope!")
        req = ToolRequest(tool_name="bad", tool_args={}, agent_name="a")
        handler = AsyncMock()

        result = await p.wrap_tool_call(req, handler)
        assert result == "Nope!"

    @pytest.mark.asyncio
    async def test_custom_deny_message_callable(self):
        p = ToolGuardPlugin(
            blocked=["bad"],
            on_deny=lambda name, args: f"Cannot run {name}",
        )
        req = ToolRequest(tool_name="bad", tool_args={"x": 1}, agent_name="a")
        handler = AsyncMock()

        result = await p.wrap_tool_call(req, handler)
        assert result == "Cannot run bad"
