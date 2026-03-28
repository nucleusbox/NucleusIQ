"""Tests for nucleusiq_gemini._shared.models (request models)."""

from nucleusiq_gemini._shared.models import (
    FileData,
    FunctionCallPart,
    FunctionResponsePart,
    GeminiContent,
    GeminiPart,
    GeminiRole,
    GenerateContentPayload,
    GenerationConfig,
    HarmBlockThreshold,
    HarmCategory,
    InlineData,
    SafetySetting,
    ThinkingConfig,
)


class TestInlineData:
    def test_construction(self):
        d = InlineData(mime_type="image/png", data="base64data==")
        assert d.mime_type == "image/png"
        assert d.data == "base64data=="


class TestFileData:
    def test_with_uri(self):
        f = FileData(file_uri="gs://bucket/file.pdf")
        assert f.file_uri == "gs://bucket/file.pdf"
        assert f.mime_type is None

    def test_with_mime(self):
        f = FileData(mime_type="application/pdf", file_uri="gs://b/f.pdf")
        assert f.mime_type == "application/pdf"


class TestFunctionCallPart:
    def test_defaults(self):
        fc = FunctionCallPart(name="fn")
        assert fc.name == "fn"
        assert fc.args == {}
        assert fc.id is None

    def test_with_args(self):
        fc = FunctionCallPart(name="fn", args={"x": 1}, id="c1")
        assert fc.args == {"x": 1}
        assert fc.id == "c1"


class TestFunctionResponsePart:
    def test_construction(self):
        fr = FunctionResponsePart(name="fn", response={"result": "ok"})
        assert fr.name == "fn"
        assert fr.response == {"result": "ok"}


class TestGeminiPart:
    def test_text_part(self):
        p = GeminiPart(text="Hello")
        d = p.to_api_dict()
        assert d == {"text": "Hello"}

    def test_inline_data_part(self):
        p = GeminiPart(inline_data=InlineData(mime_type="image/png", data="abc"))
        d = p.to_api_dict()
        assert "inline_data" in d
        assert d["inline_data"]["mime_type"] == "image/png"

    def test_function_call_part(self):
        p = GeminiPart(function_call=FunctionCallPart(name="fn", args={"x": 1}))
        d = p.to_api_dict()
        assert "function_call" in d

    def test_empty_fields_excluded(self):
        p = GeminiPart(text="Hi")
        d = p.to_api_dict()
        assert "inline_data" not in d
        assert "function_call" not in d


class TestGeminiContent:
    def test_user_message(self):
        c = GeminiContent(role="user", parts=[GeminiPart(text="Hello")])
        d = c.to_api_dict()
        assert d["role"] == "user"
        assert len(d["parts"]) == 1
        assert d["parts"][0] == {"text": "Hello"}

    def test_model_message(self):
        c = GeminiContent(role="model", parts=[GeminiPart(text="Hi!")])
        d = c.to_api_dict()
        assert d["role"] == "model"


class TestGeminiRole:
    def test_user(self):
        assert GeminiRole.USER == "user"

    def test_model(self):
        assert GeminiRole.MODEL == "model"


class TestHarmCategory:
    def test_values(self):
        assert HarmCategory.HARASSMENT == "HARM_CATEGORY_HARASSMENT"
        assert HarmCategory.HATE_SPEECH == "HARM_CATEGORY_HATE_SPEECH"
        assert HarmCategory.SEXUALLY_EXPLICIT == "HARM_CATEGORY_SEXUALLY_EXPLICIT"
        assert HarmCategory.DANGEROUS_CONTENT == "HARM_CATEGORY_DANGEROUS_CONTENT"


class TestHarmBlockThreshold:
    def test_values(self):
        assert HarmBlockThreshold.BLOCK_NONE == "BLOCK_NONE"
        assert HarmBlockThreshold.BLOCK_ONLY_HIGH == "BLOCK_ONLY_HIGH"
        assert HarmBlockThreshold.OFF == "OFF"


class TestSafetySetting:
    def test_construction(self):
        s = SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="BLOCK_ONLY_HIGH",
        )
        assert s.category == "HARM_CATEGORY_HARASSMENT"


class TestThinkingConfig:
    def test_construction(self):
        tc = ThinkingConfig(thinking_budget=1024)
        assert tc.thinking_budget == 1024

    def test_zero_budget(self):
        tc = ThinkingConfig(thinking_budget=0)
        assert tc.thinking_budget == 0


class TestGenerationConfig:
    def test_defaults(self):
        gc = GenerationConfig()
        d = gc.to_api_dict()
        assert d == {}

    def test_with_values(self):
        gc = GenerationConfig(temperature=0.5, max_output_tokens=1024, top_k=40)
        d = gc.to_api_dict()
        assert d["temperature"] == 0.5
        assert d["max_output_tokens"] == 1024
        assert d["top_k"] == 40

    def test_none_excluded(self):
        gc = GenerationConfig(temperature=0.5)
        d = gc.to_api_dict()
        assert "top_p" not in d
        assert "stop_sequences" not in d

    def test_with_thinking(self):
        gc = GenerationConfig(thinking_config=ThinkingConfig(thinking_budget=2048))
        d = gc.to_api_dict()
        assert d["thinking_config"]["thinking_budget"] == 2048

    def test_with_structured_output(self):
        gc = GenerationConfig(
            response_mime_type="application/json",
            response_json_schema={"type": "object"},
        )
        d = gc.to_api_dict()
        assert d["response_mime_type"] == "application/json"


class TestGenerateContentPayload:
    def test_build(self):
        content = GeminiContent(role="user", parts=[GeminiPart(text="Hi")])
        gen_config = GenerationConfig(temperature=0.5)
        payload = GenerateContentPayload.build(
            model="gemini-2.5-flash",
            contents=[content],
            generation_config=gen_config,
        )
        assert payload.model == "gemini-2.5-flash"
        assert len(payload.contents) == 1
        assert payload.config["temperature"] == 0.5

    def test_to_api_kwargs(self):
        content = GeminiContent(role="user", parts=[GeminiPart(text="Hi")])
        payload = GenerateContentPayload.build(
            model="gemini-2.5-flash",
            contents=[content],
        )
        kwargs = payload.to_api_kwargs()
        assert "model" in kwargs
        assert "contents" in kwargs
        assert "config" not in kwargs

    def test_with_tools(self):
        content = GeminiContent(role="user", parts=[GeminiPart(text="Hi")])
        payload = GenerateContentPayload.build(
            model="gemini-2.5-flash",
            contents=[content],
            tools=[{"function_declarations": [{"name": "fn"}]}],
        )
        assert payload.tools is not None

    def test_with_system_instruction(self):
        content = GeminiContent(role="user", parts=[GeminiPart(text="Hi")])
        payload = GenerateContentPayload.build(
            model="gemini-2.5-flash",
            contents=[content],
            system_instruction="Be helpful",
        )
        assert payload.system_instruction == "Be helpful"

    def test_with_safety(self):
        content = GeminiContent(role="user", parts=[GeminiPart(text="Hi")])
        payload = GenerateContentPayload.build(
            model="gemini-2.5-flash",
            contents=[content],
            safety_settings=[
                SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"
                )
            ],
        )
        assert payload.safety_settings is not None
        assert len(payload.safety_settings) == 1
