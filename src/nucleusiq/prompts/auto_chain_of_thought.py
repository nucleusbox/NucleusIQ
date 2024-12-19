# src/nucleusiq/prompts/auto_chain_of_thought.py

from typing import List, Dict, Optional
from nucleusiq.prompts.base import BasePrompt
from pydantic import Field
from nucleusiq.utilities.clustering import cluster_questions
from nucleusiq.llms.base_llm import BaseLLM


class AutoChainOfThoughtPrompt(BasePrompt):
    """
    Implements Automatic Chain-of-Thought (Auto-CoT) Prompting.
    Automates the generation of reasoning chains by clustering questions and generating diverse examples.
    """

    num_clusters: int = Field(default=5, description="Number of clusters to partition questions into.")
    max_questions_per_cluster: int = Field(default=1, description="Number of representative questions to sample from each cluster.")
    instruction: str = Field(default="Let's think step by step.", description="Instruction to encourage the model to generate reasoning steps.")

    # Optional variables for template placeholders
    system: Optional[str] = Field(default=None, description="System prompt including instructions.")
    context: Optional[str] = Field(default=None, description="Additional context or background information.")
    examples: Optional[str] = Field(default="", description="Generated examples based on clustered questions.")
    task_prompt: Optional[str] = Field(default="", description="Generated task prompt.")
    cot_instruction: Optional[str] = Field(default="", description="Chain-of-Thought instruction to append.")
    user: Optional[str] = Field(default=None, description="User prompt or query.")

    # Required field for the LLM adapter
    llm: BaseLLM = Field(..., description="Language Model to generate reasoning chains.")

    @property
    def technique_name(self) -> str:
        return "auto_chain_of_thought"

    @classmethod
    def default_template(cls) -> str:
        """
        Provides the default template for Auto-CoT Prompting.
        """
        return "{system}\n\n{context}\n\n{examples}\n\n{user}\n\n{cot_instruction}"

    @classmethod
    def default_input_variables(cls) -> List[str]:
        """
        Provides the default list of required input variables for Auto-CoT Prompting.
        """
        return ["task", "questions"]

    @classmethod
    def default_optional_variables(cls) -> List[str]:
        """
        Provides the default list of optional variables for Auto-CoT Prompting.
        """
        return [
            "num_clusters",
            "max_questions_per_cluster",
            "instruction",
            "system",
            "context",
            "examples",
            "task_prompt",
            "cot_instruction",
            "user"
        ]

    def construct_prompt(self, **kwargs) -> str:
        """
        Constructs an Auto-CoT prompt by generating diverse examples and appending the task prompt.

        Args:
            **kwargs: Variable values, expecting 'task', 'questions', 'system', 'context', and 'user'.

        Returns:
            str: The constructed Auto-CoT prompt.
        """
        task = kwargs.get("task")
        questions = kwargs.get("questions")
        system_prompt = kwargs.get("system", "")
        context = kwargs.get("context", "")
        user_prompt = kwargs.get("user", "")
        cot_instruction = kwargs.get("cot_instruction", self.instruction)  # Use default if not provided

        if not task or not questions:
            raise ValueError("AutoChainOfThoughtPrompt requires 'task' and 'questions' variables.")

        # Stage 1: Question Clustering
        clusters = cluster_questions(questions, num_clusters=self.num_clusters)
        representative_questions = []
        for cluster_id, cluster_questions_list in clusters.items():
            # Stage 2: Demonstration Sampling
            for q in cluster_questions_list[:self.max_questions_per_cluster]:
                representative_questions.append(q)

        # Generate reasoning chains for representative questions
        examples = []
        for question in representative_questions:
            reasoning_chain = self.generate_reasoning_chain(question)
            example_prompt = f"{question}\nA: {reasoning_chain}"
            examples.append(example_prompt)

        examples_formatted = "\n\n".join(examples)
        task_prompt = f"{task}\nA:"

        # Assign 'examples' and 'task_prompt' to kwargs for template substitution
        formatted_kwargs = {
            "system": system_prompt,
            "context": context,
            "examples": examples_formatted,
            "user": user_prompt,
            "cot_instruction": cot_instruction
        }

        # Substitute the template
        try:
            prompt = self.template.format(**formatted_kwargs)
        except KeyError as e:
            missing_key = e.args[0]
            raise ValueError(f"Missing variable for template substitution: {missing_key}")

        return prompt

    def generate_reasoning_chain(self, question: str) -> str:
        """
        Generates a reasoning chain for a given question using the provided LLM.

        Args:
            question (str): The question to generate a reasoning chain for.

        Returns:
            str: The generated reasoning chain.
        """
        if not self.llm:
            raise ValueError("Language Model (llm) is not set.")

        messages = [
            {"role": "system", "content": self.system or "You are a helpful assistant."},
            {"role": "user", "content": f"{question}\n{self.instruction}"}
        ]

        reasoning = self.llm.create_completion(
            messages=messages,
            max_tokens=150,
            temperature=0.5,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\nA:"]
        )

        return reasoning
