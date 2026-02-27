"""Additional coverage tests for tools/base_tool.py."""

from __future__ import annotations

from typing import Any

import pytest
from nucleusiq.tools import base_tool as mod
from pydantic import BaseModel


class _RefModel(BaseModel):
    kind: str


class _UsesRef(BaseModel):
    ref: _RefModel


class _CallsSuperTool(mod.BaseTool):
    async def initialize(self) -> None:
        # Intentionally call abstract base impl to cover pass.
        return await super().initialize()

    async def execute(self, **kwargs: Any) -> Any:
        await super().execute(**kwargs)
        return kwargs

    def get_spec(self):
        super().get_spec()
        return {"name": self.name, "description": self.description, "parameters": {}}


def test_pydantic_import_guard_branch(monkeypatch):
    monkeypatch.setattr(mod, "PYDANTIC_AVAILABLE", False)
    monkeypatch.setattr(mod, "BaseModel", None)
    with pytest.raises(ImportError, match="Pydantic is required"):
        mod._pydantic_model_to_json_schema(_RefModel)


def test_pydantic_schema_ref_resolution():
    schema = mod._pydantic_model_to_json_schema(_UsesRef)
    assert schema["properties"]["ref"]["type"] == "object"


@pytest.mark.asyncio
async def test_abstract_super_pass_lines_are_executable():
    t = _CallsSuperTool(name="x", description="y")
    await t.initialize()
    out = await t.execute(a=1)
    assert out["a"] == 1
    assert t.get_spec()["name"] == "x"

