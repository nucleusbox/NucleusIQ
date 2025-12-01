# src/examples/prompts/example_meta_prompt_usage.py

import os
import sys

# Add src directory to path so we can import nucleusiq
_src_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, _src_dir)

from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.prompts.meta_prompt import MetaPrompt

def initialize_meta_prompt() -> MetaPrompt:
    """
    Initializes the MetaPrompt instance with primary instructions.
    
    Returns:
        MetaPrompt: Configured MetaPrompt instance.
    """
    # Define the primary instruction for generating a blog post prompt
    primary_instruction = (
        "You are an expert technical writer. Create an effective prompt for writing a technical blog post."
    )
    
    # Define initial feedback instruction for refining the prompt
    feedback_instruction = (
        "Ensure the prompt requests inclusion of introduction, key concepts, practical examples, and a conclusion."
    )
    
    # Create and configure the MetaPrompt instance
    meta_prompt = (
        PromptFactory
        .create_prompt(PromptTechnique.META_PROMPTING)
        .configure(
            primary_instruction=primary_instruction,
            feedback_instruction=feedback_instruction
        )
    )
    
    return meta_prompt

# Initialize the MetaPrompt
meta_prompt = initialize_meta_prompt()

def generate_initial_blog_prompt(meta_prompt: MetaPrompt) -> str:
    """
    Generates the initial prompt for writing a technical blog post.
    
    Args:
        meta_prompt (MetaPrompt): The MetaPrompt instance.
    
    Returns:
        str: The generated prompt.
    """
    # Generate the initial prompt using MetaPrompt
    initial_prompt = meta_prompt.format_prompt(
        generated_prompt="Write a detailed technical blog post about the latest advancements in renewable energy."
    )
    
    print("=== Initial Generated Blog Prompt ===")
    print(initial_prompt)
    print("\n")
    
    return initial_prompt

# Generate and display the initial blog prompt
initial_blog_prompt = generate_initial_blog_prompt(meta_prompt)

def refine_blog_prompt(meta_prompt: MetaPrompt, current_prompt: str) -> str:
    """
    Refines the generated blog post prompt based on feedback.
    
    Args:
        meta_prompt (MetaPrompt): The MetaPrompt instance.
        current_prompt (str): The current generated prompt.
    
    Returns:
        str: The refined prompt.
    """
    # Define feedback to improve the prompt
    feedback = (
        "Emphasize the inclusion of practical examples and real-world applications to enhance understanding."
    )
    
    # Refine the prompt using the feedback
    refined_prompt = meta_prompt.refine_prompt(
        feedback=feedback,
        current_prompt=current_prompt
    )
    
    print("=== Refined Blog Prompt ===")
    print(refined_prompt)
    print("\n")
    
    return refined_prompt

# Refine the generated blog prompt based on feedback
refined_blog_prompt = refine_blog_prompt(meta_prompt, initial_blog_prompt)

def generate_blog_post(meta_prompt: MetaPrompt, topic: str) -> str:
    """
    Generates a technical blog post using the refined prompt.
    
    Args:
        meta_prompt (MetaPrompt): The MetaPrompt instance.
        topic (str): The topic of the blog post.
    
    Returns:
        str: The generated blog post.
    """
    # Generate the blog prompt
    blog_prompt = meta_prompt.format_prompt(
        generated_prompt=f"Write a detailed technical blog post about {topic}."
    )
    
    print("=== Generated Blog Prompt ===")
    print(blog_prompt)
    print("\n")
    
    # Simulate sending the prompt to an LLM (e.g., OpenAI's GPT-4)
    # Here, we'll mock the LLM's response
    llm_response = mock_llm_response(blog_prompt, topic)
    
    # Optionally, apply the output parser if defined
    if meta_prompt.output_parser:
        parsed_output = meta_prompt.output_parser(llm_response)
    else:
        parsed_output = llm_response
    
    print("=== Generated Blog Post ===")
    print(parsed_output)
    print("\n")
    
    return parsed_output

def mock_llm_response(prompt: str, topic: str) -> str:
    """
    Mocks the response from an LLM based on the given prompt and topic.
    
    Args:
        prompt (str): The prompt sent to the LLM.
        topic (str): The topic of the blog post.
    
    Returns:
        str: Mocked blog post content.
    """
    # Mocked response for demonstration purposes
    return (
        f"### {topic.capitalize()}\n\n"
        "Renewable energy has seen significant advancements in recent years, driven by technological innovations and increasing environmental awareness. "
        "This blog post explores the latest developments in renewable energy, highlighting key concepts, practical examples, and real-world applications.\n\n"
        "#### Introduction\n\n"
        "Renewable energy sources, such as solar, wind, and hydroelectric power, play a crucial role in reducing carbon emissions and combating climate change. "
        "Advancements in these technologies have made renewable energy more efficient and accessible than ever before.\n\n"
        "#### Key Concepts\n\n"
        "1. **Solar Photovoltaics (PV)**: Improvements in solar panel efficiency and storage solutions have made solar energy a more viable option for both residential and commercial use.\n"
        "2. **Wind Turbines**: The development of larger and more efficient wind turbines has increased energy output while reducing costs.\n"
        "3. **Energy Storage**: Enhanced battery technologies are enabling better storage of renewable energy, ensuring a steady supply even when production fluctuates.\n\n"
        "#### Practical Examples\n\n"
        "1. **Tesla's Powerwall**: This home battery system stores energy from solar panels, providing backup power during outages and optimizing energy usage.\n"
        "2. **GE Renewable Energy's Offshore Wind Farms**: These large-scale wind farms are generating substantial amounts of electricity, contributing significantly to national grids.\n\n"
        "#### Conclusion\n\n"
        "The advancements in renewable energy technologies are paving the way for a more sustainable and environmentally friendly future. "
        "Continued innovation and investment in this sector are essential for meeting global energy demands while preserving our planet."
    )

# Generate and display the technical blog post
topic = "the latest advancements in renewable energy"
blog_post = generate_blog_post(meta_prompt, topic)

def further_refine_blog_prompt(meta_prompt: MetaPrompt, refined_prompt: str) -> str:
    """
    Further refines the generated blog post prompt based on additional feedback.
    
    Args:
        meta_prompt (MetaPrompt): The MetaPrompt instance.
        refined_prompt (str): The current refined prompt.
    
    Returns:
        str: The further refined prompt.
    """
    # Define additional feedback to enhance the prompt
    additional_feedback = (
        "Ensure that the prompt requests detailed explanations of the methodologies and technologies behind the advancements."
    )
    
    # Refine the prompt using the additional feedback
    further_refined_prompt = meta_prompt.refine_prompt(
        feedback=additional_feedback,
        current_prompt=refined_prompt
    )
    
    print("=== Further Refined Blog Prompt ===")
    print(further_refined_prompt)
    print("\n")
    
    return further_refined_prompt

# Apply further refinement based on additional feedback
further_refined_blog_prompt = further_refine_blog_prompt(meta_prompt, refined_blog_prompt)

def generate_enhanced_blog_post(meta_prompt: MetaPrompt, topic: str) -> str:
    """
    Generates an enhanced technical blog post using the further refined prompt.
    
    Args:
        meta_prompt (MetaPrompt): The MetaPrompt instance.
        topic (str): The topic of the blog post.
    
    Returns:
        str: The generated blog post.
    """
    # Generate the refined blog prompt
    blog_prompt = meta_prompt.format_prompt(
        generated_prompt=f"Write a detailed technical blog post about {topic}."
    )
    
    print("=== Generated Enhanced Blog Prompt ===")
    print(blog_prompt)
    print("\n")
    
    # Simulate sending the prompt to an LLM (e.g., OpenAI's GPT-4)
    # Here, we'll mock the LLM's response
    llm_response = mock_llm_response_enhanced(blog_prompt, topic)
    
    # Optionally, apply the output parser if defined
    if meta_prompt.output_parser:
        parsed_output = meta_prompt.output_parser(llm_response)
    else:
        parsed_output = llm_response
    
    print("=== Generated Enhanced Blog Post ===")
    print(parsed_output)
    print("\n")
    
    return parsed_output

def mock_llm_response_enhanced(prompt: str, topic: str) -> str:
    """
    Mocks the response from an LLM based on the given prompt and topic, including methodologies.
    
    Args:
        prompt (str): The prompt sent to the LLM.
        topic (str): The topic of the blog post.
    
    Returns:
        str: Mocked blog post content with detailed methodologies.
    """
    # Mocked response for demonstration purposes
    return (
        f"### {topic.capitalize()}\n\n"
        "Renewable energy has witnessed remarkable advancements in recent years, driven by cutting-edge technologies and heightened environmental consciousness. "
        "This blog post delves into the latest developments in renewable energy, elucidating key concepts, practical examples, detailed methodologies, and real-world applications.\n\n"
        "#### Introduction\n\n"
        "Renewable energy sources like solar, wind, and hydroelectric power are pivotal in mitigating climate change by reducing greenhouse gas emissions. "
        "Technological innovations have enhanced the efficiency, scalability, and affordability of these energy sources, making them more accessible to a broader population.\n\n"
        "#### Key Concepts\n\n"
        "1. **Solar Photovoltaics (PV)**: Recent advancements in photovoltaic cell technology have significantly increased solar panel efficiency. "
        "Multi-junction cells and perovskite materials are at the forefront, enabling higher energy conversion rates and reduced production costs.\n"
        "2. **Wind Turbines**: The development of larger, more aerodynamic turbine blades and offshore wind farms has boosted energy output. "
        "Innovations like floating wind turbines allow for installation in deeper waters, expanding the potential for wind energy generation.\n"
        "3. **Energy Storage**: Enhanced battery technologies, including lithium-ion and flow batteries, have improved energy storage capacity and lifespan. "
        "These advancements ensure a steady energy supply, even when renewable sources are intermittent.\n\n"
        "#### Methodologies and Technologies\n\n"
        "1. **Differential Privacy in Data Handling**: To optimize energy distribution, differential privacy techniques are employed to analyze usage patterns without compromising individual data privacy.\n"
        "2. **Machine Learning for Predictive Maintenance**: AI-driven predictive maintenance systems monitor equipment health, predicting failures before they occur and minimizing downtime.\n"
        "3. **Smart Grid Integration**: Smart grids utilize IoT devices and advanced analytics to manage energy distribution efficiently, balancing supply and demand in real-time.\n\n"
        "#### Practical Examples\n\n"
        "1. **Tesla's Solar Roof**: Combining aesthetics with functionality, Tesla's Solar Roof integrates solar cells seamlessly into roof tiles, providing both energy generation and structural integrity.\n"
        "2. **GE Renewable Energy's Offshore Wind Farms**: These installations harness strong offshore winds, generating substantial electricity to support national grids.\n"
        "3. **Energy Vault's Gravity-Based Storage**: This innovative storage solution uses excess energy to lift massive weights, storing potential energy that can be released when needed.\n\n"
        "#### Conclusion\n\n"
        "The continuous advancements in renewable energy technologies are crucial for achieving a sustainable and environmentally friendly future. "
        "By embracing innovations in solar PV, wind turbines, energy storage, and smart grid integration, we can significantly reduce our carbon footprint and ensure energy security for generations to come. "
        "Ongoing research and investment in these areas will further accelerate the transition towards a green economy."
    )

# Generate and display the enhanced technical blog post
enhanced_blog_post = generate_enhanced_blog_post(meta_prompt, topic)

def format_section_header(section_title: str) -> str:
    """
    Formats a section header in Markdown.
    
    Args:
        section_title (str): The title of the section.
    
    Returns:
        str: Formatted Markdown header.
    """
    return f"#### {section_title}\n\n"

def generate_summary(paragraph: str) -> str:
    """
    Generates a concise summary of a given paragraph.
    
    Args:
        paragraph (str): The paragraph to summarize.
    
    Returns:
        str: Summary of the paragraph.
    """
    # Mock summary for demonstration
    return paragraph[:150] + "..."

def initialize_meta_prompt_with_functions() -> MetaPrompt:
    """
    Initializes the MetaPrompt instance with primary instructions and function mappings.
    
    Returns:
        MetaPrompt: Configured MetaPrompt instance with function mappings.
    """
    primary_instruction = (
        "You are an expert technical writer. Create an effective prompt for writing a technical blog post."
    )
    
    feedback_instruction = (
        "Ensure the prompt requests inclusion of introduction, key concepts, practical examples, methodologies, technologies, and a conclusion."
    )
    
    # Define function mappings
    func_mappings = {
        "format_section_header": format_section_header,
        "generate_summary": generate_summary
    }
    
    # Create and configure the MetaPrompt instance
    meta_prompt = (
        PromptFactory
        .create_prompt(PromptTechnique.META_PROMPTING)
        .configure(
            primary_instruction=primary_instruction,
            feedback_instruction=feedback_instruction,
            function_mappings=func_mappings
        )
    )
    
    return meta_prompt

# Initialize the MetaPrompt with function mappings
meta_prompt_functions = initialize_meta_prompt_with_functions()

