# src/nucleusiq/prompts/auto_chain_of_thought.py

from typing import List, Dict, Optional, Any
from pydantic import Field, field_validator
from nucleusiq.prompts.base import BasePrompt
from nucleusiq.llms.base_llm import BaseLLM

class AutoChainOfThoughtPrompt(BasePrompt):
    """
    Implements Automatic Chain-of-Thought (Auto-CoT) Prompting.
    Automates question clustering and reasoning chain generation.
    """

    # Primary fields controlling clustering / CoT
    num_clusters: int = Field(
        default=5,
        description="Number of clusters for question grouping (>=1)."
    )
    max_questions_per_cluster: int = Field(
        default=1,
        description="Number of questions to pick per cluster in each cluster (>=1)."
    )
    instruction: str = Field(
        default="Let's think step by step.",
        description="Default CoT instruction if user doesn't supply a custom one."
    )

    # We'll consider 'task' + 'questions' as required input_variables
    template: str = Field(
        default="{system}\n\n{context}\n\n{task}\n\n{examples}\n\n{user}\n\n{cot_instruction}",
        description="Default template for Auto-CoT Prompting."
    )
    input_variables: List[str] = Field(
        default_factory=lambda: ["task", "questions"],
        description="Mandatory fields for Auto-CoT: 'task' (str) and 'questions' (list)."
    )
    optional_variables: List[str] = Field(
        default_factory=lambda: [
            "system",
            "context",
            "examples",
            "user",
            "cot_instruction",
            "task_prompt",
            "num_clusters",
            "max_questions_per_cluster",
            "instruction",
            "llm"  # We'll check usage of llm in a hook
        ],
        description="Other optional fields recognized by the base class."
    )

    # Subclass fields (some optional)
    system: Optional[str] = None
    context: Optional[str] = None
    task_prompt: Optional[str] = Field(default="", description="Optional final task text if desired.")
    user: Optional[str] = None
    examples: Optional[str] = Field(default="", description="Auto-generated examples text.")
    cot_instruction: Optional[str] = Field(default="", description="Extra CoT text appended at the end.")
    llm: Optional[BaseLLM] = Field(
        default=None,
        description="Language Model used to generate reasoning chains (must be non-None to run)."
    )

    @property
    def technique_name(self) -> str:
        return "auto_chain_of_thought"

    #
    # Numeric validations
    #
    @field_validator("num_clusters")
    def validate_num_clusters(cls, v):
        if v < 1:
            raise ValueError("num_clusters must be >= 1.")
        return v

    @field_validator("max_questions_per_cluster")
    def validate_max_questions_per_cluster(cls, v):
        if v < 1:
            raise ValueError("max_questions_per_cluster must be >= 1.")
        return v

    #
    # Subclass-specific checks
    #
    def _pre_format_validation(self, combined_vars: Dict[str, Any]) -> None:
        """
        Ensures 'llm' is set, 'task' is non-empty, 'questions' is a non-empty list, etc.
        """
        if self.llm is None:
            raise ValueError("AutoChainOfThoughtPrompt requires 'llm' to be set (non-None).")

        # Check 'task' is a non-empty string
        task_val = combined_vars.get("task", "")
        if not isinstance(task_val, str) or not task_val.strip():
            raise ValueError("AutoChainOfThoughtPrompt requires 'task' be a non-empty string.")

        # Check 'questions' is a non-empty list
        questions_val = combined_vars.get("questions", [])
        if not isinstance(questions_val, list) or len(questions_val) == 0:
            raise ValueError("AutoChainOfThoughtPrompt requires a non-empty list of 'questions'.")

    def _construct_prompt(self, **kwargs) -> str:
        """
        1) Cluster the questions => generate reasoning chain => build final text
        2) Insert system, context, task, examples, user, CoT text
        """
        task_text = kwargs["task"]
        questions = kwargs["questions"]

        from nucleusiq.utilities.clustering import cluster_questions
        clusters = cluster_questions(questions, num_clusters=self.num_clusters)

        # Collect representative questions
        rep_questions = []
        for cluster_list in clusters.values():
            rep_questions.extend(cluster_list[: self.max_questions_per_cluster])

        # Generate chain text for each question
        example_lines = []
        for q in rep_questions:
            chain = self.generate_reasoning_chain(q)
            example_lines.append(f"{q}\nA: {chain}")

        final_examples = "\n\n".join(example_lines)

        # Merge other fields
        system_prompt = kwargs.get("system", "") or self.system or ""
        context_str = kwargs.get("context", "") or self.context or ""
        user_prompt = kwargs.get("user", "") or self.user or ""
        cot_text = kwargs.get("cot_instruction", "") or self.cot_instruction
        if not cot_text.strip() and self.instruction.strip():
            cot_text = self.instruction

        parts = []
        if system_prompt.strip():
            parts.append(system_prompt.strip())
        if context_str.strip():
            parts.append(context_str.strip())

        # Now insert the mandatory 'task' text as well
        if task_text.strip():
            parts.append(task_text.strip())

        if final_examples.strip():
            parts.append(final_examples.strip())

        if user_prompt.strip():
            parts.append(user_prompt.strip())

        if cot_text.strip():
            parts.append(cot_text.strip())

        return "\n\n".join(parts)

    def generate_reasoning_chain(self, question: str) -> str:
        """
        Calls self.llm to produce a reasoning chain for the question.
        """
        if not self.llm:
            raise ValueError("llm not set. Please assign an LLM instance before usage.")

        sys_prompt = self.system or "You are a helpful assistant."
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": f"{question}\n{self.instruction}"}
        ]
        return self.llm.create_completion(
            messages=messages,
            max_tokens=150,
            temperature=0.5,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\nA:"]
        )
