# examples/retrieval_augmented_generation_example.py

from nucleusiq.prompts.factory import PromptFactory, PromptTechnique




# examples/retrieval_augmented_generation_example.py

from nucleusiq.prompts.factory import PromptFactory, PromptTechnique

def rag_example():
    rag = PromptFactory.create_prompt(PromptTechnique.RETRIEVAL_AUGMENTED_GENERATION)
    # Set multiple parameters at once
    rag.set_parameters(
        system="You are a specialized research assistant.",
        context="Quantum computing breakthroughs from 2023 ...",
        user="Summarize the latest quantum computing breakthroughs."
    )
    # Format the final prompt
    prompt_text = rag.format_prompt()
    print(prompt_text)

    # Now set the fields you need
    rag_prompt = PromptFactory.create_prompt(
    PromptTechnique.RETRIEVAL_AUGMENTED_GENERATION
)
    rag_prompt.system = "You are an expert researcher."
    rag_prompt.user = "Provide a summary of the latest advancements in quantum computing."
    rag_prompt.context = """
    Quantum computing has seen significant advancements in error correction, ...
    """
    formatted_rag = rag_prompt.format_prompt()
    print(formatted_rag)


if __name__ == "__main__":
    rag_example()

