"""Tests covering gaps in prompts/auto_chain_of_thought.py, base.py, factory.py."""

import json

import pytest
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.prompts.auto_chain_of_thought import AutoChainOfThoughtPrompt
from nucleusiq.prompts.chain_of_thought import ChainOfThoughtPrompt
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.prompts.few_shot import FewShotPrompt
from nucleusiq.prompts.zero_shot import ZeroShotPrompt

# ═══════════════════════════════════════════════════════════════════════════════
# AutoChainOfThoughtPrompt
# ═══════════════════════════════════════════════════════════════════════════════


class TestAutoChainOfThought:
    def test_technique_name(self):
        p = AutoChainOfThoughtPrompt(llm=MockLLM())
        assert p.technique_name == "auto_chain_of_thought"

    def test_num_clusters_validation(self):
        with pytest.raises(ValueError, match="num_clusters"):
            AutoChainOfThoughtPrompt(num_clusters=0, llm=MockLLM())

    def test_max_questions_validation(self):
        with pytest.raises(ValueError, match="max_questions_per_cluster"):
            AutoChainOfThoughtPrompt(max_questions_per_cluster=0, llm=MockLLM())

    def test_no_llm_raises(self):
        p = AutoChainOfThoughtPrompt()
        with pytest.raises(ValueError, match="llm"):
            p.format_prompt(task="Do X", questions=["Q1"])

    def test_empty_task_raises(self):
        p = AutoChainOfThoughtPrompt(llm=MockLLM())
        with pytest.raises(ValueError, match="task"):
            p.format_prompt(task="", questions=["Q1"])

    def test_empty_questions_raises(self):
        p = AutoChainOfThoughtPrompt(llm=MockLLM())
        with pytest.raises(ValueError, match="questions"):
            p.format_prompt(task="Do X", questions=[])

    def test_format_prompt_basic(self):
        llm = MockLLM()
        p = AutoChainOfThoughtPrompt(
            llm=llm,
            num_clusters=1,
            max_questions_per_cluster=1,
        )
        questions = [
            "How does photosynthesis work in plants?",
            "What are the stages of cellular respiration?",
            "Explain the water cycle and its importance.",
            "How do enzymes catalyze biochemical reactions?",
            "What is the role of DNA in protein synthesis?",
        ]
        result = p.format_prompt(task="Explain biology concepts", questions=questions)
        assert isinstance(result, str)
        assert "biology" in result.lower()

    def test_generate_reasoning_chain_no_llm(self):
        p = AutoChainOfThoughtPrompt()
        with pytest.raises(ValueError, match="llm"):
            p.generate_reasoning_chain("Q?")

    def test_generate_reasoning_chain(self):
        llm = MockLLM()
        p = AutoChainOfThoughtPrompt(llm=llm, system="Be smart")
        chain = p.generate_reasoning_chain("Why is the sky blue?")
        assert isinstance(chain, str)


# ═══════════════════════════════════════════════════════════════════════════════
# PromptFactory
# ═══════════════════════════════════════════════════════════════════════════════


class TestPromptFactory:
    def test_all_techniques_registered(self):
        for tech in PromptTechnique:
            assert tech.value in PromptFactory.prompt_classes

    def test_create_zero_shot(self):
        p = PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT)
        assert isinstance(p, ZeroShotPrompt)

    def test_create_chain_of_thought(self):
        p = PromptFactory.create_prompt(PromptTechnique.CHAIN_OF_THOUGHT)
        assert isinstance(p, ChainOfThoughtPrompt)

    def test_create_few_shot(self):
        p = PromptFactory.create_prompt(PromptTechnique.FEW_SHOT)
        assert isinstance(p, FewShotPrompt)

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            PromptFactory.register_prompt(PromptTechnique.ZERO_SHOT, ZeroShotPrompt)


# ═══════════════════════════════════════════════════════════════════════════════
# BasePrompt save / load
# ═══════════════════════════════════════════════════════════════════════════════


class TestBasePromptSaveLoad:
    def test_save_load_json(self, tmp_path):
        p = ZeroShotPrompt()
        p.configure(system="Test system", user="Test user")
        path = tmp_path / "prompt.json"
        p.save(path)
        loaded = ZeroShotPrompt.load(path)
        assert loaded.system == "Test system"
        assert loaded.user == "Test user"

    def test_save_load_yaml(self, tmp_path):
        p = ZeroShotPrompt()
        p.configure(system="S", user="U")
        path = tmp_path / "prompt.yaml"
        p.save(path)
        loaded = ZeroShotPrompt.load(path)
        assert loaded.system == "S"

    def test_save_unsupported_format(self, tmp_path):
        p = ZeroShotPrompt()
        with pytest.raises(ValueError, match="Unsupported"):
            p.save(tmp_path / "prompt.xml")

    def test_load_unsupported_format(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported"):
            ZeroShotPrompt.load(tmp_path / "prompt.toml")

    def test_load_missing_type(self, tmp_path):
        path = tmp_path / "no_type.json"
        path.write_text(json.dumps({"system": "x"}))
        with pytest.raises(ValueError, match="_type"):
            ZeroShotPrompt.load(path)

    def test_load_unknown_type(self, tmp_path):
        path = tmp_path / "unknown.json"
        path.write_text(json.dumps({"_type": "nonexistent_technique"}))
        with pytest.raises(ValueError, match="Unsupported prompt type"):
            ZeroShotPrompt.load(path)


# ═══════════════════════════════════════════════════════════════════════════════
# BasePrompt additional methods
# ═══════════════════════════════════════════════════════════════════════════════


class TestBasePromptMethods:
    def test_set_metadata(self):
        p = ZeroShotPrompt()
        p.set_metadata({"author": "test"})
        assert p.metadata["author"] == "test"

    def test_add_tags(self):
        p = ZeroShotPrompt()
        p.add_tags(["tag1", "tag2"])
        assert "tag1" in p.tags

    def test_set_output_parser(self):
        p = ZeroShotPrompt()
        parser = lambda x: x.upper()
        p.set_output_parser(parser)
        assert p.output_parser is parser
