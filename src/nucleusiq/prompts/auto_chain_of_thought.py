# src/nucleusiq/prompts/auto_chain_of_thought.py

from typing import List, Dict, Optional
from pydantic import Field
from nucleusiq.prompts.base import BasePrompt
from nucleusiq.llms.base_llm import BaseLLM
from nucleusiq.utilities.clustering import cluster_questions


class AutoChainOfThoughtPrompt(BasePrompt):
    """
    Implements Automatic Chain-of-Thought (Auto-CoT) Prompting.
    Automates the generation of reasoning chains by clustering questions and generating diverse examples.
    """

    # Main CoT fields
    num_clusters: int = Field(
        default=5,
        description="Number of clusters to partition questions into."
    )
    max_questions_per_cluster: int = Field(
        default=1,
        description="Number of representative questions to sample from each cluster."
    )
    instruction: str = Field(
        default="Let's think step by step.",
        description="Instruction to encourage the model to generate reasoning steps."
    )

    # Template + input/optional variables
    template: str = Field(
        default_factory=lambda: AutoChainOfThoughtPrompt.default_template()
    )
    input_variables: List[str] = Field(
        default_factory=lambda: AutoChainOfThoughtPrompt.default_input_variables()
    )
    optional_variables: List[str] = Field(
        default_factory=lambda: AutoChainOfThoughtPrompt.default_optional_variables()
    )

    # Optional placeholders in the template
    # system: Optional[str] = Field(default=None, description="System prompt or instructions.")
    # context: Optional[str] = Field(default=None, description="Additional context or background info.")
    examples: Optional[str] = Field(default="", description="Generated examples after question clustering.")
    # user: Optional[str] = Field(default=None, description="User prompt or query.")
    cot_instruction: Optional[str] = Field(default="", description="Additional CoT instruction appended.")
    task_prompt: Optional[str] = Field(default="", description="Optional field if you want a final task prompt text.")

    # LLM is optional at construction but required at usage
    llm: Optional[BaseLLM] = Field(
        default=None,
        description="Language Model used to generate reasoning chains. Must be set before usage."
    )

    @property
    def technique_name(self) -> str:
        return "auto_chain_of_thought"

    @staticmethod
    def default_template() -> str:
        """
        Provides the default template for Auto-CoT Prompting.
        Note: The template isn't strictly used with the list+join approach,
        but we keep it for consistency with BasePrompt requirements.
        """
        return "{system}\n\n{context}\n\n{examples}\n\n{user}\n\n{cot_instruction}"

    @staticmethod
    def default_input_variables() -> List[str]:
        """List of required inputs (e.g. 'task' and 'questions')."""
        return ["task", "questions"]

    @staticmethod
    def default_optional_variables() -> List[str]:
        """List of optional fields recognized at runtime."""
        return [
            "num_clusters",
            "max_questions_per_cluster",
            "instruction",
            "system",
            "context",
            "examples",
            "cot_instruction",
            "task_prompt",
            "user",
        ]

    def set_parameters(
        self,
        system: Optional[str] = None,
        context: Optional[str] = None,
        user: Optional[str] = None,
        instruction: Optional[str] = None,
        num_clusters: Optional[int] = None,
        max_questions_per_cluster: Optional[int] = None,
        cot_instruction: Optional[str] = None,
    ) -> "AutoChainOfThoughtPrompt":
        """
        Sets parameters for the AutoCoT prompt in one go.

        Args:
            system: System instructions or role.
            context: Additional background info.
            user: The user question or request.
            instruction: The default CoT instruction (e.g. "Let's think step by step.").
            num_clusters: # of clusters for question grouping.
            max_questions_per_cluster: # of questions to pick per cluster.
            cot_instruction: Additional CoT text appended at the end.

        Returns:
            AutoChainOfThoughtPrompt: This instance with updated fields.
        """
        if system is not None:
            self.system = system
        if context is not None:
            self.context = context
        if user is not None:
            self.user = user
        if instruction is not None:
            self.instruction = instruction
        if num_clusters is not None:
            self.num_clusters = num_clusters
        if max_questions_per_cluster is not None:
            self.max_questions_per_cluster = max_questions_per_cluster
        if cot_instruction is not None:
            self.cot_instruction = cot_instruction
        return self

    def _construct_prompt(self, **kwargs) -> str:
        """
        Constructs an Auto-CoT prompt by:
          1) Clustering questions and generating reasoning examples.
          2) Merging system, context, generated examples, user, and CoT into a list.
          3) Only appending non-empty parts to avoid extra blank lines.
        Required: 'task' (str) and 'questions' (List[str]) at runtime.
        """
        # 1) Required inputs
        task = kwargs.get("task")
        questions = kwargs.get("questions")
        if not task or not questions:
            raise ValueError("AutoChainOfThoughtPrompt requires 'task' and 'questions' at runtime.")
        if not isinstance(questions, list):
            raise ValueError("'questions' must be a list of strings.")

        # 2) Generate the 'examples' text from questions
        from nucleusiq.utilities.clustering import cluster_questions  # or your logic
        clusters = cluster_questions(questions, num_clusters=self.num_clusters)
        rep_questions = []
        for cluster_id, cluster_list in clusters.items():
            rep_questions.extend(cluster_list[: self.max_questions_per_cluster])

        # Build example text by calling generate_reasoning_chain
        example_strs = []
        for q in rep_questions:
            chain = self.generate_reasoning_chain(q)
            example_strs.append(f"{q}\nA: {chain}")
        final_examples = "\n\n".join(example_strs)

        # 3) Gather the final strings for system, context, user, cot
        system_prompt = kwargs.get("system", "") or self.system or ""
        context_str = kwargs.get("context", "") or self.context or ""
        user_prompt = kwargs.get("user", "") or self.user or ""
        cot_text = kwargs.get("cot_instruction", self.instruction) or self.cot_instruction

        # If final_examples is empty, we skip it; otherwise we use it
        parts = []
        if system_prompt.strip():
            parts.append(system_prompt)
        if context_str.strip():
            parts.append(context_str)
        if final_examples.strip():
            parts.append(final_examples)
        if user_prompt.strip():
            parts.append(user_prompt)
        if cot_text.strip():
            parts.append(cot_text)

        # 4) Return the joined result with \n\n
        return "\n\n".join(parts)

    def generate_reasoning_chain(self, question: str) -> str:
        """
        Uses the assigned LLM to generate a reasoning chain for 'question'.
        Raises ValueError if llm is not set.
        """
        if not self.llm:
            raise ValueError("llm is not set. Please assign an LLM instance before usage.")

        # If no system assigned, default to "You are a helpful assistant."
        sys_text = self.system or "You are a helpful assistant."
        messages = [
            {"role": "system", "content": sys_text},
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
