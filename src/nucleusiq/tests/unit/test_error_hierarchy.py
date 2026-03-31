"""Tests for the unified NucleusIQ exception hierarchy (v0.7.2).

Verifies:
- All exceptions inherit from NucleusIQError
- Re-parented existing errors (PluginError, StructuredOutputError, etc.)
- Structured context attributes on every error class
- Single catch-all works: ``except NucleusIQError``
"""

import pytest
from nucleusiq.agents.errors import (
    AgentConfigError,
    AgentError,
    AgentExecutionError,
    AgentTimeoutError,
    AttachmentError,
    AttachmentProcessingError,
    AttachmentUnsupportedError,
    AttachmentValidationError,
)
from nucleusiq.agents.structured_output.errors import (
    MultipleOutputError,
    SchemaParseError,
    SchemaValidationError,
    StructuredOutputError,
)
from nucleusiq.errors import NucleusIQError
from nucleusiq.llms.errors import (
    AuthenticationError,
    ContentFilterError,
    ContextLengthError,
    InvalidRequestError,
    LLMError,
    ModelNotFoundError,
    PermissionDeniedError,
    ProviderConnectionError,
    ProviderError,
    ProviderServerError,
    RateLimitError,
)
from nucleusiq.memory.errors import (
    MemoryCapacityError,
    MemoryImportError,
    MemoryReadError,
    MemoryWriteError,
    NucleusMemoryError,
)
from nucleusiq.plugins.errors import (
    PluginError,
    PluginExecutionError,
    PluginHalt,
)
from nucleusiq.prompts.errors import (
    PromptConfigError,
    PromptError,
    PromptGenerationError,
    PromptTemplateError,
)
from nucleusiq.streaming.errors import (
    StreamingError,
    StreamInterruptedError,
    StreamOrchestrationError,
)
from nucleusiq.tools.builtin.workspace import WorkspaceSecurityError
from nucleusiq.tools.errors import (
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolPermissionError,
    ToolTimeoutError,
    ToolValidationError,
)


class TestNucleusIQErrorRoot:
    def test_is_base_exception(self):
        assert issubclass(NucleusIQError, Exception)

    def test_catch_all(self):
        with pytest.raises(NucleusIQError):
            raise LLMError("test")

    def test_catch_all_tool(self):
        with pytest.raises(NucleusIQError):
            raise ToolExecutionError("fail")

    def test_catch_all_plugin(self):
        with pytest.raises(NucleusIQError):
            raise PluginHalt("halt")


class TestLLMErrorHierarchy:
    def test_base_inherits_from_root(self):
        assert issubclass(LLMError, NucleusIQError)

    def test_all_subtypes_inherit(self):
        for cls in [
            AuthenticationError,
            PermissionDeniedError,
            RateLimitError,
            InvalidRequestError,
            ModelNotFoundError,
            ContentFilterError,
            ContextLengthError,
            ProviderServerError,
            ProviderConnectionError,
            ProviderError,
        ]:
            assert issubclass(cls, LLMError), f"{cls.__name__} not subclass of LLMError"
            assert issubclass(cls, NucleusIQError)

    def test_attributes(self):
        e = RateLimitError(
            "Too many requests",
            provider="openai",
            status_code=429,
        )
        assert e.provider == "openai"
        assert e.status_code == 429
        assert e.original_error is None
        assert str(e) == "Too many requests"

    def test_context_length_error(self):
        e = ContextLengthError(
            "Input exceeds 128k",
            provider="openai",
            status_code=400,
        )
        assert isinstance(e, LLMError)
        assert e.provider == "openai"


class TestToolErrorHierarchy:
    def test_base_inherits_from_root(self):
        assert issubclass(ToolError, NucleusIQError)

    def test_all_subtypes(self):
        for cls in [
            ToolExecutionError,
            ToolTimeoutError,
            ToolValidationError,
            ToolPermissionError,
            ToolNotFoundError,
        ]:
            assert issubclass(cls, ToolError)
            assert issubclass(cls, NucleusIQError)

    def test_attributes(self):
        e = ToolExecutionError(
            "Division by zero",
            tool_name="calculator",
            original_error=ZeroDivisionError("division by zero"),
            args_snapshot={"a": 1, "b": 0},
        )
        assert e.tool_name == "calculator"
        assert isinstance(e.original_error, ZeroDivisionError)
        assert e.args_snapshot == {"a": 1, "b": 0}


class TestAgentErrorHierarchy:
    def test_base_inherits_from_root(self):
        assert issubclass(AgentError, NucleusIQError)

    def test_all_subtypes(self):
        for cls in [AgentConfigError, AgentExecutionError, AgentTimeoutError]:
            assert issubclass(cls, AgentError)

    def test_attributes(self):
        e = AgentConfigError(
            "Missing LLM",
            mode="standard",
            task_id="t1",
        )
        assert e.mode == "standard"
        assert e.task_id == "t1"


class TestAttachmentErrorHierarchy:
    def test_base_inherits_from_root(self):
        assert issubclass(AttachmentError, NucleusIQError)

    def test_all_subtypes(self):
        for cls in [
            AttachmentValidationError,
            AttachmentProcessingError,
            AttachmentUnsupportedError,
        ]:
            assert issubclass(cls, AttachmentError)

    def test_attributes(self):
        e = AttachmentValidationError(
            "Too large",
            attachment_type="image_url",
            file_name="photo.jpg",
        )
        assert e.attachment_type == "image_url"
        assert e.file_name == "photo.jpg"


class TestMemoryErrorHierarchy:
    def test_base_inherits_from_root(self):
        assert issubclass(NucleusMemoryError, NucleusIQError)

    def test_all_subtypes(self):
        for cls in [
            MemoryWriteError,
            MemoryReadError,
            MemoryImportError,
            MemoryCapacityError,
        ]:
            assert issubclass(cls, NucleusMemoryError)

    def test_does_not_shadow_builtin(self):
        assert NucleusMemoryError is not MemoryError


class TestPromptErrorHierarchy:
    def test_base_inherits_from_root(self):
        assert issubclass(PromptError, NucleusIQError)

    def test_all_subtypes(self):
        for cls in [PromptTemplateError, PromptConfigError, PromptGenerationError]:
            assert issubclass(cls, PromptError)

    def test_attributes(self):
        e = PromptTemplateError(
            "Missing variable {name}",
            technique="chain_of_thought",
            template_name="cot_v2",
        )
        assert e.technique == "chain_of_thought"
        assert e.template_name == "cot_v2"


class TestStreamingErrorHierarchy:
    def test_base_inherits_from_root(self):
        assert issubclass(StreamingError, NucleusIQError)

    def test_all_subtypes(self):
        for cls in [StreamInterruptedError, StreamOrchestrationError]:
            assert issubclass(cls, StreamingError)


class TestReparentedErrors:
    def test_plugin_error_under_root(self):
        assert issubclass(PluginError, NucleusIQError)

    def test_plugin_halt_under_plugin_error(self):
        assert issubclass(PluginHalt, PluginError)

    def test_plugin_execution_error(self):
        assert issubclass(PluginExecutionError, PluginError)

    def test_structured_output_under_root(self):
        assert issubclass(StructuredOutputError, NucleusIQError)

    def test_structured_output_subtypes(self):
        for cls in [SchemaValidationError, SchemaParseError, MultipleOutputError]:
            assert issubclass(cls, StructuredOutputError)

    def test_workspace_security_under_root(self):
        assert issubclass(WorkspaceSecurityError, NucleusIQError)

    def test_workspace_security_attributes(self):
        e = WorkspaceSecurityError(
            "Path escape",
            path="/etc/passwd",
            workspace_root="/workspace",
        )
        assert e.path == "/etc/passwd"
        assert e.workspace_root == "/workspace"

    def test_plugin_halt_result_attribute(self):
        e = PluginHalt("early_result")
        assert e.result == "early_result"
        assert isinstance(e, NucleusIQError)
