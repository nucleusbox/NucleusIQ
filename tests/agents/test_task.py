"""
Comprehensive tests for Task class.

Tests cover:
- Task creation (positive scenarios)
- Task validation (negative scenarios)
- Task serialization/deserialization
- Task from_dict conversion
- Edge cases and error handling
"""

import os
import sys
from pathlib import Path

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest
import uuid
from typing import Dict, Any
from pydantic import ValidationError

from nucleusiq.agents.task import Task


class TestTaskCreation:
    """Test Task creation with various scenarios."""
    
    def test_task_creation_minimal(self):
        """Test creating task with only required fields (id and objective)."""
        task = Task(id="task1", objective="What is 5 + 3?")
        
        assert task.objective == "What is 5 + 3?"
        assert task.id == "task1"
        assert task.context is None
        assert task.metadata is None
    
    def test_task_creation_with_all_fields(self):
        """Test creating task with all fields."""
        task = Task(
            id="task1",
            objective="Calculate 15 + 27",
            context={"user": "test_user", "session": "session123"},
            metadata={"priority": "high", "source": "api"}
        )
        
        assert task.id == "task1"
        assert task.objective == "Calculate 15 + 27"
        assert task.context == {"user": "test_user", "session": "session123"}
        assert task.metadata == {"priority": "high", "source": "api"}
    
    def test_task_creation_with_context_only(self):
        """Test creating task with context but no metadata."""
        task = Task(
            id="task1",
            objective="Test task",
            context={"key": "value"}
        )
        
        assert task.objective == "Test task"
        assert task.context == {"key": "value"}
        assert task.metadata is None
    
    def test_task_creation_with_metadata_only(self):
        """Test creating task with metadata but no context."""
        task = Task(
            id="task1",
            objective="Test task",
            metadata={"key": "value"}
        )
        
        assert task.objective == "Test task"
        assert task.context is None
        assert task.metadata == {"key": "value"}
    
    def test_task_id_required(self):
        """Test that task ID is required (no auto-generation)."""
        # ID is required, so this should fail
        with pytest.raises(ValidationError):
            Task(objective="Task 1")  # type: ignore[call-arg]
        
        # But we can provide it
        task1 = Task(id="task1", objective="Task 1")
        task2 = Task(id="task2", objective="Task 2")
        
        assert task1.id == "task1"
        assert task2.id == "task2"
    
    def test_task_with_empty_string_objective(self):
        """Test creating task with empty string objective."""
        # Empty string is allowed by Pydantic (unless we add validation)
        task = Task(id="task1", objective="")
        assert task.objective == ""
    
    def test_task_with_none_objective(self):
        """Test creating task with None objective (should fail)."""
        with pytest.raises(ValidationError):
            Task(id="task1", objective=None)  # type: ignore[arg-type]
    
    def test_task_with_whitespace_only_objective(self):
        """Test creating task with whitespace-only objective."""
        task = Task(id="task1", objective="   ")
        assert task.objective == "   "  # Pydantic doesn't strip by default
    
    def test_task_with_complex_context(self):
        """Test creating task with complex nested context."""
        task = Task(
            id="task1",
            objective="Complex task",
            context={
                "nested": {
                    "level1": {
                        "level2": "value"
                    }
                },
                "list": [1, 2, 3],
                "mixed": {"key": "value", "number": 42}
            }
        )
        
        assert task.context is not None
        assert task.context["nested"]["level1"]["level2"] == "value"
        assert task.context["list"] == [1, 2, 3]
        assert task.context["mixed"]["number"] == 42
    
    def test_task_with_complex_metadata(self):
        """Test creating task with complex nested metadata."""
        task = Task(
            id="task1",
            objective="Complex task",
            metadata={
                "tags": ["urgent", "important"],
                "config": {
                    "timeout": 30,
                    "retries": 3
                }
            }
        )
        
        assert task.metadata is not None
        assert "urgent" in task.metadata["tags"]
        assert task.metadata["config"]["timeout"] == 30


class TestTaskSerialization:
    """Test Task serialization and deserialization."""
    
    def test_task_to_dict(self):
        """Test converting task to dictionary."""
        task = Task(
            id="task1",
            objective="Test objective",
            context={"key": "value"},
            metadata={"meta": "data"}
        )
        
        task_dict = task.to_dict()
        
        assert isinstance(task_dict, dict)
        assert task_dict["id"] == "task1"
        assert task_dict["objective"] == "Test objective"
        assert task_dict["context"] == {"key": "value"}
        assert task_dict["metadata"] == {"meta": "data"}
    
    def test_task_to_dict_minimal(self):
        """Test converting minimal task to dictionary."""
        task = Task(id="task1", objective="Minimal task")
        task_dict = task.to_dict()
        
        assert task_dict["objective"] == "Minimal task"
        assert task_dict["context"] is None
        assert task_dict["metadata"] is None
        assert task_dict["id"] == "task1"
    
    def test_task_from_dict(self):
        """Test creating task from dictionary."""
        task_dict = {
            "id": "task1",
            "objective": "Test objective",
            "context": {"key": "value"},
            "metadata": {"meta": "data"}
        }
        
        task = Task.from_dict(task_dict)
        
        assert task.id == "task1"
        assert task.objective == "Test objective"
        assert task.context == {"key": "value"}
        assert task.metadata == {"meta": "data"}
    
    def test_task_from_dict_minimal(self):
        """Test creating task from minimal dictionary (requires id)."""
        task_dict = {
            "id": "task1",
            "objective": "Minimal task"
        }
        
        task = Task.from_dict(task_dict)
        
        assert task.objective == "Minimal task"
        assert task.context is None
        assert task.metadata is None
        assert task.id == "task1"
    
    def test_task_from_dict_with_id_string(self):
        """Test creating task from dict with ID as string."""
        task_id_str = str(uuid.uuid4())
        task_dict = {
            "id": task_id_str,
            "objective": "Test task"
        }
        
        task = Task.from_dict(task_dict)
        
        assert str(task.id) == task_id_str
    
    def test_task_from_dict_with_id_uuid(self):
        """Test creating task from dict with ID as UUID object (converted to string)."""
        task_id = uuid.uuid4()
        task_dict = {
            "id": str(task_id),  # Convert to string first
            "objective": "Test task"
        }
        
        # UUID string should work
        task = Task.from_dict(task_dict)
        
        assert task.id == str(task_id)
    
    def test_task_round_trip_serialization(self):
        """Test serializing and deserializing task maintains data."""
        original_task = Task(
            id="task1",
            objective="Round trip test",
            context={"test": "data"},
            metadata={"meta": "info"}
        )
        
        task_dict = original_task.to_dict()
        restored_task = Task.from_dict(task_dict)
        
        assert restored_task.id == original_task.id
        assert restored_task.objective == original_task.objective
        assert restored_task.context == original_task.context
        assert restored_task.metadata == original_task.metadata


class TestTaskValidation:
    """Test Task validation and error handling."""
    
    def test_task_missing_objective(self):
        """Test that missing objective raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Task()  # type: ignore[call-arg]
        
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("objective",) for error in errors)
    
    def test_task_id_as_string(self):
        """Test that ID can be any string (not just UUID)."""
        # ID is just a string, so any string is valid
        task = Task(id="any-string-id", objective="Test task")
        assert task.id == "any-string-id"
    
    def test_task_invalid_context_type(self):
        """Test that invalid context type raises ValidationError."""
        with pytest.raises(ValidationError):
            Task(
                id="task1",
                objective="Test task",
                context="not-a-dict"  # type: ignore[arg-type]
            )
    
    def test_task_invalid_metadata_type(self):
        """Test that invalid metadata type raises ValidationError."""
        with pytest.raises(ValidationError):
            Task(
                id="task1",
                objective="Test task",
                metadata="not-a-dict"  # type: ignore[arg-type]
            )
    
    def test_task_from_dict_invalid_objective(self):
        """Test from_dict with invalid objective."""
        with pytest.raises(ValidationError):
            Task.from_dict({"objective": None})
    
    def test_task_from_dict_missing_objective(self):
        """Test from_dict with missing objective."""
        with pytest.raises(ValidationError):
            Task.from_dict({})
    
    def test_task_from_dict_extra_fields(self):
        """Test from_dict with extra fields (should be ignored)."""
        # Pydantic by default ignores extra fields
        task_dict = {
            "id": "task1",
            "objective": "Test task",
            "extra_field": "should be ignored"
        }
        
        task = Task.from_dict(task_dict)
        assert task.objective == "Test task"
        # Extra field should not be in task
        assert not hasattr(task, "extra_field")


class TestTaskEdgeCases:
    """Test Task edge cases and special scenarios."""
    
    def test_task_with_very_long_objective(self):
        """Test task with very long objective string."""
        long_objective = "A" * 10000
        task = Task(id="task1", objective=long_objective)
        
        assert len(task.objective) == 10000
        assert task.objective == long_objective
    
    def test_task_with_special_characters(self):
        """Test task with special characters in objective."""
        special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        task = Task(id="task1", objective=f"Test {special_chars}")
        
        assert special_chars in task.objective
    
    def test_task_with_unicode_characters(self):
        """Test task with unicode characters."""
        unicode_obj = "测试任务 🚀 日本語"
        task = Task(id="task1", objective=unicode_obj)
        
        assert task.objective == unicode_obj
    
    def test_task_with_none_context(self):
        """Test task with None context."""
        task = Task(id="task1", objective="Test", context=None)
        
        # None is allowed (it's Optional)
        assert task.context is None
    
    def test_task_with_none_metadata(self):
        """Test task with None metadata."""
        task = Task(id="task1", objective="Test", metadata=None)
        
        # None is allowed (it's Optional)
        assert task.metadata is None
    
    def test_task_context_immutability(self):
        """Test that modifying context dict doesn't affect task."""
        original_context = {"key": "value"}
        task = Task(id="task1", objective="Test", context=original_context)
        
        # Modify original dict
        original_context["new_key"] = "new_value"
        
        # Task context should not change (Pydantic creates a copy)
        assert task.context is not None and "new_key" not in task.context
    
    def test_task_metadata_immutability(self):
        """Test that modifying metadata dict doesn't affect task."""
        original_metadata = {"key": "value"}
        task = Task(id="task1", objective="Test", metadata=original_metadata)
        
        # Modify original dict
        original_metadata["new_key"] = "new_value"
        
        # Task metadata should not change (Pydantic creates a copy)
        assert task.metadata is not None and "new_key" not in task.metadata


class TestTaskComparison:
    """Test Task comparison and equality."""
    
    def test_task_equality_same_id(self):
        """Test that tasks with same ID have same id."""
        task1 = Task(id="task1", objective="Task 1")
        task2 = Task(id="task1", objective="Task 2")
        
        # Tasks with same ID
        assert task1.id == task2.id
    
    def test_task_inequality_different_id(self):
        """Test that tasks with different IDs have different ids."""
        task1 = Task(id="task1", objective="Task 1")
        task2 = Task(id="task2", objective="Task 1")
        
        # Different IDs
        assert task1.id != task2.id


class TestTaskIntegration:
    """Test Task integration with other components."""
    
    @pytest.mark.asyncio
    async def test_task_with_agent_execute(self):
        """Test task can be used with agent.execute()."""
        from nucleusiq.agents import Agent
        from nucleusiq.core.llms.mock_llm import MockLLM
        
        llm = MockLLM()
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            llm=llm
        )
        
        task = Task(id="task1", objective="What is 2 + 2?")
        
        # Should not raise error
        result = await agent.execute(task)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_task_dict_compatibility(self):
        """Test task works with dict-based agent methods."""
        from nucleusiq.agents import Agent
        from nucleusiq.core.llms.mock_llm import MockLLM
        
        llm = MockLLM()
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            llm=llm
        )
        
        task = Task(id="task1", objective="What is 2 + 2?")
        task_dict = task.to_dict()
        
        # Should work with dict
        result = await agent.execute(task_dict)
        assert result is not None

