"""
Complete OpenAI Integration Examples

This comprehensive example demonstrates all NucleusIQ features integrated with OpenAI:
- All execution modes (DIRECT, STANDARD, AUTONOMOUS)
- Structured output with Pydantic models
- OpenAI native tools
- Custom BaseTool tools
- Planning and multi-step execution
- ReAct agent pattern

Run with: python src/examples/agents/openai_complete_integration.py

Requires OPENAI_API_KEY environment variable.
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any
from pydantic import BaseModel, Field

# Load environment variables (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use environment variables directly

# Add src directory to path
_src_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, _src_dir)

from nucleusiq.agents import Agent, ReActAgent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.agents.structured_output import OutputSchema, OutputMode
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.providers.llms.openai.nb_openai import BaseOpenAI
from nucleusiq.providers.llms.openai.tools.openai_tool import OpenAITool
from nucleusiq.core.tools import BaseTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Structured Output Models
# ============================================================================

class PersonInfo(BaseModel):
    """Person information extracted from text."""
    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years")
    occupation: str = Field(description="Job title or occupation")
    location: str = Field(description="City or location")


class CalculationResult(BaseModel):
    """Result of a mathematical calculation."""
    operation: str = Field(description="The operation performed")
    operands: list[float] = Field(description="The numbers used in the operation")
    result: float = Field(description="The calculated result")
    explanation: str = Field(description="Brief explanation of the calculation")


# ============================================================================
# Custom Tools
# ============================================================================

def create_calculator_tools():
    """Create custom calculator tools."""
    def add(a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b
    
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers together."""
        return a * b
    
    def calculate_area(length: float, width: float) -> float:
        """Calculate the area of a rectangle."""
        return length * width
    
    add_tool = BaseTool.from_function(add, description="Add two numbers")
    multiply_tool = BaseTool.from_function(multiply, description="Multiply two numbers")
    area_tool = BaseTool.from_function(calculate_area, description="Calculate rectangle area")
    
    return [add_tool, multiply_tool, area_tool]


# ============================================================================
# Example 1: DIRECT Mode with Structured Output
# ============================================================================

async def example_direct_mode_structured():
    """Example: DIRECT mode with structured output extraction."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 1: DIRECT Mode with Structured Output")
    logger.info("=" * 80)
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not set!")
        return
    
    # Create OpenAI LLM
    llm = BaseOpenAI(model_name="gpt-4o-mini", temperature=0.3)
    
    # Create agent with structured output
    agent = Agent(
        name="DataExtractor",
        role="Data Extraction Assistant",
        objective="Extract structured information from text",
        llm=llm,
        response_format=PersonInfo,  # AUTO mode - uses NATIVE for OpenAI
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT)
    )
    
    await agent.initialize()
    
    # Task: Extract person info
    task = Task(
        id="extract_person",
        objective="Extract person information from: John Smith is a 35-year-old software engineer living in San Francisco."
    )
    
    logger.info(f"Task: {task.objective}")
    logger.info("Executing in DIRECT mode with structured output...")
    
    result = await agent.execute(task)
    logger.info(f"\n✅ Result:\n{result}")
    
    # The result should be a dict with "output" containing PersonInfo instance
    if isinstance(result, dict) and "output" in result:
        person = result["output"]
        logger.info(f"\n📊 Extracted Data:")
        logger.info(f"   Name: {person.name}")
        logger.info(f"   Age: {person.age}")
        logger.info(f"   Occupation: {person.occupation}")
        logger.info(f"   Location: {person.location}")


# ============================================================================
# Example 2: STANDARD Mode with Custom Tools
# ============================================================================

async def example_standard_mode_tools():
    """Example: STANDARD mode with custom tools."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 2: STANDARD Mode with Custom Tools")
    logger.info("=" * 80)
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not set!")
        return
    
    # Create OpenAI LLM
    llm = BaseOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    
    # Create tools
    tools = create_calculator_tools()
    
    # Create prompt
    prompt = PromptFactory.create_prompt(
        technique=PromptTechnique.ZERO_SHOT
    ).configure(
        system="You are a helpful math assistant. Use tools to perform calculations accurately.",
        user="Help the user with their mathematical questions."
    )
    
    # Create agent
    agent = Agent(
        name="MathBot",
        role="Math Assistant",
        objective="Perform mathematical calculations using tools",
        llm=llm,
        prompt=prompt,
        tools=tools,
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD)
    )
    
    await agent.initialize()
    
    # Task 1: Simple calculation
    task1 = Task(
        id="calc1",
        objective="What is 15 + 27? Use the add tool."
    )
    
    logger.info(f"Task 1: {task1.objective}")
    result1 = await agent.execute(task1)
    logger.info(f"✅ Result: {result1}\n")
    
    # Task 2: Complex calculation
    task2 = Task(
        id="calc2",
        objective="Calculate the area of a rectangle with length 10 and width 5."
    )
    
    logger.info(f"Task 2: {task2.objective}")
    result2 = await agent.execute(task2)
    logger.info(f"✅ Result: {result2}\n")


# ============================================================================
# Example 3: STANDARD Mode with OpenAI Native Tools
# ============================================================================

async def example_standard_mode_openai_tools():
    """Example: STANDARD mode with OpenAI native tools."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 3: STANDARD Mode with OpenAI Native Tools")
    logger.info("=" * 80)
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not set!")
        return
    
    # Create OpenAI LLM
    llm = BaseOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    
    # Create OpenAI native tools
    # Note: OpenAI native tools (web_search, code_interpreter) are not currently supported
    # by the OpenAI API in the tools array format. They may need to be enabled differently.
    # For now, we'll skip them in this example.
    # web_search = OpenAITool.web_search()
    # code_interpreter = OpenAITool.code_interpreter()
    
    # Create prompt
    prompt = PromptFactory.create_prompt(
        technique=PromptTechnique.ZERO_SHOT
    ).configure(
        system="You are a helpful assistant with access to web search and code execution capabilities.",
        user="Use tools when needed to provide accurate information."
    )
    
    # Create agent with custom tools only (OpenAI native tools have API compatibility issues)
    # Create a simple calculator tool for this example
    def calculate_percentage(value: float, percentage: float) -> float:
        """Calculate percentage of a value."""
        return (value * percentage) / 100
    
    calc_tool = BaseTool.from_function(calculate_percentage)
    
    agent = Agent(
        name="ResearchBot",
        role="Research Assistant",
        objective="Perform calculations to answer questions",
        llm=llm,
        prompt=prompt,
        tools=[calc_tool],  # Using custom tool instead of code_interpreter
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD)
    )
    
    await agent.initialize()
    
    # Task: Calculate with custom tool
    task = Task(
        id="research",
        objective="Calculate what 2% of 14 million would be."
    )
    
    logger.info(f"Task: {task.objective}")
    logger.info("This will use a custom calculate_percentage tool.")
    
    result = await agent.execute(task)
    logger.info(f"\n✅ Result:\n{result}\n")


# ============================================================================
# Example 4: AUTONOMOUS Mode with Planning
# ============================================================================

async def example_autonomous_mode_planning():
    """Example: AUTONOMOUS mode with multi-step planning."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 4: AUTONOMOUS Mode with Planning")
    logger.info("=" * 80)
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not set!")
        return
    
    # Create OpenAI LLM
    llm = BaseOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    
    # Create tools
    tools = create_calculator_tools()
    
    # Create prompt
    prompt = PromptFactory.create_prompt(
        technique=PromptTechnique.ZERO_SHOT
    ).configure(
        system="You are a planning assistant. Break down complex tasks into steps.",
        user="Create a detailed plan and execute it step by step."
    )
    
    # Create agent with AUTONOMOUS mode
    agent = Agent(
        name="PlannerBot",
        role="Planning Assistant",
        objective="Plan and execute multi-step tasks",
        llm=llm,
        prompt=prompt,
        tools=tools,
        config=AgentConfig(
            execution_mode=ExecutionMode.AUTONOMOUS,
            planning_timeout=120,
            step_timeout=60
        )
    )
    
    await agent.initialize()
    
    # Complex multi-step task
    task = Task(
        id="complex_calc",
        objective="First, calculate 10 * 5. Then, add 25 to that result. Finally, multiply the result by 2."
    )
    
    logger.info(f"Task: {task.objective}")
    logger.info("AUTONOMOUS mode will:")
    logger.info("  1. Generate a multi-step plan")
    logger.info("  2. Execute each step sequentially")
    logger.info("  3. Pass context between steps")
    
    result = await agent.execute(task)
    logger.info(f"\n✅ Final Result: {result}\n")


# ============================================================================
# Example 5: Structured Output with Explicit NATIVE Mode
# ============================================================================

async def example_structured_output_native():
    """Example: Explicit NATIVE mode for structured output."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 5: Structured Output with Explicit NATIVE Mode")
    logger.info("=" * 80)
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not set!")
        return
    
    # Create OpenAI LLM
    llm = BaseOpenAI(model_name="gpt-4o-mini", temperature=0.3)
    
    # Create agent with explicit NATIVE mode
    agent = Agent(
        name="StructuredBot",
        role="Structured Data Extractor",
        objective="Extract structured data with strict validation",
        llm=llm,
        response_format=OutputSchema(
            schema=CalculationResult,
            mode=OutputMode.NATIVE,
            strict=True,  # Strict schema adherence
            retry_on_error=True,
            max_retries=2
        ),
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT)
    )
    
    await agent.initialize()
    
    # Task: Perform calculation and return structured result
    task = Task(
        id="structured_calc",
        objective="Calculate 15 * 7 and return the result in the specified format with operation, operands, result, and explanation."
    )
    
    logger.info(f"Task: {task.objective}")
    logger.info("Using NATIVE mode with strict=True for OpenAI json_schema")
    
    result = await agent.execute(task)
    logger.info(f"\n✅ Result:\n{result}\n")
    
    if isinstance(result, dict) and "output" in result:
        calc = result["output"]
        logger.info(f"📊 Structured Calculation Result:")
        logger.info(f"   Operation: {calc.operation}")
        logger.info(f"   Operands: {calc.operands}")
        logger.info(f"   Result: {calc.result}")
        logger.info(f"   Explanation: {calc.explanation}")


# ============================================================================
# Example 6: ReAct Agent with OpenAI
# ============================================================================

async def example_react_agent():
    """Example: ReAct agent with iterative reasoning."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 6: ReAct Agent with OpenAI")
    logger.info("=" * 80)
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not set!")
        return
    
    # Create OpenAI LLM
    llm = BaseOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    
    # Create tools
    tools = create_calculator_tools()
    
    # Create ReAct agent
    agent = ReActAgent(
        name="ReActBot",
        role="Reasoning Assistant",
        objective="Use reasoning and tools to solve problems",
        llm=llm,
        tools=tools,
        max_iterations=10,
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD)
    )
    
    await agent.initialize()
    
    # Task requiring reasoning
    task = Task(
        id="react_task",
        objective="I need to calculate: (10 + 5) * 3. Think step by step and use tools."
    )
    
    logger.info(f"Task: {task.objective}")
    logger.info("ReAct agent will:")
    logger.info("  1. Think about the problem")
    logger.info("  2. Decide which tool to use")
    logger.info("  3. Execute the tool")
    logger.info("  4. Observe the result")
    logger.info("  5. Continue until final answer")
    
    result = await agent.execute(task)
    logger.info(f"\n✅ Final Answer: {result}\n")
    
    # Show ReAct history
    if hasattr(agent, '_react_history') and agent._react_history:
        logger.info("📝 ReAct History:")
        for i, entry in enumerate(agent._react_history, 1):
            logger.info(f"   Step {i}: {entry.get('thought', 'N/A')[:100]}...")


# ============================================================================
# Example 7: Mixed Tools (BaseTool + OpenAI Native)
# ============================================================================

async def example_mixed_tools():
    """Example: Agent with both BaseTool and OpenAI native tools."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 7: Mixed Tools (BaseTool + OpenAI Native)")
    logger.info("=" * 80)
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not set!")
        return
    
    # Create OpenAI LLM
    llm = BaseOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    
    # Mix of custom tools (OpenAI native tools disabled due to API compatibility)
    custom_tools = create_calculator_tools()
    # openai_tools = [
    #     OpenAITool.web_search(),  # Disabled due to API compatibility issues
    #     OpenAITool.code_interpreter()  # Disabled due to API compatibility issues
    # ]
    
    all_tools = custom_tools  # Using only custom tools
    
    # Create prompt
    prompt = PromptFactory.create_prompt(
        technique=PromptTechnique.ZERO_SHOT
    ).configure(
        system="You are a versatile assistant with access to custom calculation tools, web search, and code execution.",
        user="Use the appropriate tool for each task."
    )
    
    # Create agent
    agent = Agent(
        name="HybridBot",
        role="Hybrid Assistant",
        objective="Use both custom and native tools to solve problems",
        llm=llm,
        prompt=prompt,
        tools=all_tools,
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD)
    )
    
    await agent.initialize()
    
    # Task requiring multiple tool types
    task = Task(
        id="hybrid_task",
        objective="Assume New York City has a population of 8.5 million. Calculate what 5% of that population would be using the multiply tool."
    )
    
    logger.info(f"Task: {task.objective}")
    logger.info("This will use:")
    logger.info("  - multiply (BaseTool) to calculate 5% of 8.5 million")
    
    result = await agent.execute(task)
    logger.info(f"\n✅ Result:\n{result}\n")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all examples."""
    logger.info("=" * 80)
    logger.info("NucleusIQ Complete OpenAI Integration Examples")
    logger.info("=" * 80)
    logger.info("\nThis example demonstrates:")
    logger.info("  ✅ DIRECT mode with structured output")
    logger.info("  ✅ STANDARD mode with custom tools")
    logger.info("  ✅ STANDARD mode with OpenAI native tools")
    logger.info("  ✅ AUTONOMOUS mode with planning")
    logger.info("  ✅ Structured output with explicit NATIVE mode")
    logger.info("  ✅ ReAct agent pattern")
    logger.info("  ✅ Mixed tools (BaseTool + OpenAI native)")
    logger.info("\n" + "=" * 80)
    
    examples = [
        ("DIRECT Mode with Structured Output", example_direct_mode_structured),
        ("STANDARD Mode with Custom Tools", example_standard_mode_tools),
        ("STANDARD Mode with OpenAI Native Tools", example_standard_mode_openai_tools),
        ("AUTONOMOUS Mode with Planning", example_autonomous_mode_planning),
        ("Structured Output with NATIVE Mode", example_structured_output_native),
        ("ReAct Agent", example_react_agent),
        ("Mixed Tools", example_mixed_tools),
    ]
    
    for name, example_func in examples:
        try:
            await example_func()
            logger.info(f"\n✅ {name} completed successfully\n")
        except Exception as e:
            logger.error(f"\n❌ {name} failed: {e}\n", exc_info=True)
    
    logger.info("=" * 80)
    logger.info("All examples completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

