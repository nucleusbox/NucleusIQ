#!/usr/bin/env python3
"""
NucleusIQ Complex Autonomous Agent Examples
============================================

This example demonstrates sophisticated AUTONOMOUS mode capabilities
with MULTIPLE PROMPT TECHNIQUES:

1. Research Agent - CHAIN_OF_THOUGHT prompting (step-by-step reasoning)
2. Data Pipeline Agent - FEW_SHOT prompting (learning from examples)
3. Project Planning Agent - CHAIN_OF_THOUGHT prompting (logical planning)
4. Code Quality Agent - FEW_SHOT + CoT hybrid (examples with reasoning)
5. Strategic Decision Agent - AUTO_CHAIN_OF_THOUGHT (automatic reasoning)

Prompt Techniques Demonstrated:
- ZERO_SHOT: Direct task execution without examples
- FEW_SHOT: Learning from provided input-output examples
- CHAIN_OF_THOUGHT: Explicit step-by-step reasoning
- AUTO_CHAIN_OF_THOUGHT: Automatic reasoning chain generation

Each example showcases how AUTONOMOUS mode:
- Automatically generates execution plans
- Handles dependencies between steps
- Builds context across the workflow
- Uses different prompting strategies for different tasks
"""

import asyncio
import logging
import os
import sys
from typing import Any, Dict
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path for imports
_src_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, _src_dir)

from nucleusiq.agents.agent import Agent
from nucleusiq.agents.config.agent_config import AgentConfig, ExecutionMode, AgentState
from nucleusiq.agents.task import Task
from nucleusiq.tools import BaseTool
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq_openai import BaseOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Keep third-party logs quiet
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


# =============================================================================
# TOOL DEFINITIONS - Rich set of tools for complex tasks
# =============================================================================

def create_research_tools():
    """Create tools for research and information gathering."""
    
    def web_search(query: str) -> str:
        """Search the web for information on a topic. Returns relevant snippets."""
        # Simulated search results based on query keywords
        results = {
            "python performance": """
Top Results:
1. Python 3.12 Performance Guide - 25% faster than 3.11
2. Using Cython for 100x speedup on numerical code
3. PyPy vs CPython: When to use each
4. Async IO patterns for high-concurrency applications
Key Insight: Python 3.12+ offers significant performance improvements.""",
            "machine learning": """
Top Results:
1. Transformer Architecture Explained - Attention is All You Need
2. Fine-tuning LLMs: Best Practices 2024
3. MLOps Pipeline Design Patterns
4. Comparing PyTorch vs TensorFlow vs JAX
Key Insight: Transformers dominate modern ML architectures.""",
            "startup funding": """
Top Results:
1. Seed Round Benchmarks 2024: $2-4M average
2. Series A Requirements: $1M ARR minimum
3. VC Investment Trends: AI/ML leading sectors
4. Bootstrap vs Funded: Pros and Cons Analysis
Key Insight: AI startups receiving highest valuations.""",
            "climate data": """
Top Results:
1. Global Temperature Rise: +1.2C since pre-industrial
2. CO2 Levels: 420 ppm (March 2024)
3. Renewable Energy: 30% of global electricity
4. EV Adoption: 14% of new car sales globally
Key Insight: Climate action accelerating but gaps remain.""",
        }
        
        # Find matching results
        for key, value in results.items():
            if key.lower() in query.lower():
                return value.strip()
        
        return f"Search results for '{query}': Multiple articles found. Key topics: trends, analysis, insights available."
    
    def database_query(table: str, query_type: str = "summary") -> str:
        """Query the database for structured data. Table: users, sales, metrics."""
        mock_data = {
            "users": {
                "total_count": 15420,
                "active_last_30_days": 8234,
                "premium_users": 1245,
                "churn_rate": "3.4%",
                "growth_rate": "+12% MoM"
            },
            "sales": {
                "total_revenue": "$1,250,000",
                "monthly_growth": "12%",
                "avg_order_value": "$85.50",
                "top_product": "Enterprise Plan",
                "conversion_rate": "3.2%"
            },
            "metrics": {
                "page_views": "2,500,000",
                "unique_visitors": "450,000",
                "avg_session_duration": "4min 5sec",
                "bounce_rate": "42%",
                "engagement_score": "7.2/10"
            }
        }
        
        if table.lower() in mock_data:
            data = mock_data[table.lower()]
            return f"Database Query Result ({table}):\n" + "\n".join(f"  - {k}: {v}" for k, v in data.items())
        return f"Query on {table}: No data found. Available tables: users, sales, metrics."
    
    def analyze_data(data: str, analysis_type: str = "summary") -> str:
        """Perform statistical analysis. Types: summary, trend, correlation, forecast."""
        analyses = {
            "summary": "Statistical Summary: Mean=125.4, Median=118, StdDev=23.5, Range=[45, 234], Sample Size=1000",
            "trend": "Trend Analysis: Upward trend detected (+12% MoM), seasonality present (Q4 peaks), confidence=92%",
            "correlation": "Correlation Analysis: revenue~users=0.87 (strong), churn~satisfaction=-0.72 (negative)",
            "forecast": "Forecast: Next quarter projected at $1.4M (+12%), confidence interval: $1.25M-$1.55M (85% CI)"
        }
        return analyses.get(analysis_type.lower(), f"Analysis of {data}: Patterns identified, insights generated.")
    
    def summarize(text: str, style: str = "concise") -> str:
        """Summarize information. Styles: concise, executive, detailed."""
        word_count = len(text.split())
        styles = {
            "concise": f"Summary ({word_count} words condensed): Key themes extracted, core insights identified, actionable items highlighted.",
            "executive": f"Executive Summary: Decision points identified. Recommendation: Proceed with Option A. Risk: Moderate. Timeline: 8 weeks.",
            "detailed": f"Detailed Analysis: Comprehensive review of {word_count} word input. Multiple patterns identified. See sections below."
        }
        return styles.get(style.lower(), styles["concise"])
    
    def generate_report(content: str, title: str = "Report", format: str = "markdown") -> str:
        """Generate a formatted report. Formats: markdown, json, text."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        if format.lower() == "markdown":
            return f"""# {title}
Generated: {timestamp}

## Executive Summary
{content[:200]}...

## Key Findings
- Primary insight derived from analysis
- Secondary patterns and trends identified
- Actionable recommendations provided

## Conclusion
Based on comprehensive analysis, strategic recommendations outlined above."""
        elif format.lower() == "json":
            return f'{{"title": "{title}", "timestamp": "{timestamp}", "summary": "{content[:100]}..."}}'
        else:
            return f"Report: {title}\nDate: {timestamp}\nContent: {content[:200]}..."
    
    return [
        BaseTool.from_function(web_search, description="Search the web for information on any topic."),
        BaseTool.from_function(database_query, description="Query database tables: users, sales, metrics."),
        BaseTool.from_function(analyze_data, description="Perform statistical analysis: summary, trend, correlation, forecast."),
        BaseTool.from_function(summarize, description="Summarize text in concise, executive, or detailed style."),
        BaseTool.from_function(generate_report, description="Generate formatted reports in markdown, json, or text format."),
    ]


def create_project_tools():
    """Create tools for project planning and management."""
    
    def plan_project(description: str, team_size: int = 3) -> str:
        """Break down a project into phases, tasks, and estimates."""
        return f"""Project Plan: {description[:50]}...

Team Size: {team_size} developers
Total Duration: 8 weeks

Phase 1: Foundation (Week 1-2)
  - Task 1.1: Requirements Analysis [8h]
  - Task 1.2: Architecture Design [16h]
  - Task 1.3: Environment Setup [4h]

Phase 2: Core Development (Week 3-6)
  - Task 2.1: Backend API [40h] (depends: 1.2)
  - Task 2.2: Database Schema [16h] (depends: 1.1)
  - Task 2.3: Frontend Components [32h] (depends: 1.2)
  - Task 2.4: Integration [24h] (depends: 2.1, 2.2)

Phase 3: Quality & Launch (Week 7-8)
  - Task 3.1: Testing [24h] (depends: 2.4)
  - Task 3.2: Documentation [16h]
  - Task 3.3: Deployment [8h] (depends: 3.1)

Total Effort: 188 hours
Critical Path: 1.1 -> 1.2 -> 2.1 -> 2.4 -> 3.1 -> 3.3"""
    
    def assess_risk(scenario: str, context: str = "") -> str:
        """Assess risks and provide mitigation strategies."""
        return f"""Risk Assessment: {scenario[:40]}...

HIGH PRIORITY:
[!] Technical Risk: Integration complexity
    Probability: 60%, Impact: High
    Mitigation: Early prototyping, incremental integration

MEDIUM PRIORITY:
[*] Schedule Risk: Scope creep
    Probability: 75%, Impact: Medium
    Mitigation: Strict change control, regular reviews

[*] Resource Risk: Key person dependency
    Probability: 40%, Impact: Medium
    Mitigation: Knowledge sharing, documentation

LOW PRIORITY:
[-] External Risk: Vendor delays
    Probability: 20%, Impact: Low
    Mitigation: Alternative suppliers identified

Overall Risk Score: 6.5/10 (Moderate)
Recommendation: Proceed with enhanced monitoring"""
    
    def calculate(expression: str, calc_type: str = "basic") -> str:
        """Perform calculations: basic math, financial (roi, npv), estimates."""
        try:
            if calc_type == "basic":
                allowed = set("0123456789+-*/.() ")
                if all(c in allowed for c in expression):
                    result = eval(expression)
                    return f"Calculation: {expression} = {result}"
            elif calc_type == "roi":
                return "ROI Analysis: Investment=$100K, Return=$150K, ROI=50%, Payback=18 months"
            elif calc_type == "npv":
                return "NPV Calculation: Cash flows at 10% discount rate, NPV=$234,567, IRR=18.5%"
            elif calc_type == "estimate":
                return f"Estimate for '{expression}': 40-60 hours (most likely: 48h), $7,200-$9,000 at $150/hr"
        except Exception:
            pass
        return f"Calculated: {expression} (type: {calc_type})"
    
    def analyze_code(path: str, analysis_type: str = "full") -> str:
        """Analyze code for quality, complexity, and issues."""
        return f"""Code Analysis: {path}

Metrics:
- Lines of Code: 1,245
- Cyclomatic Complexity: 12 (moderate)
- Code Coverage: 78%
- Technical Debt: 4.5 hours

Issues Found:
- 3 security vulnerabilities (2 medium, 1 low)
- 12 code style violations
- 2 potential memory leaks
- 5 unused imports

Quality Score: 7.2/10
Recommendations:
1. Refactor complex functions (>20 lines)
2. Add error handling to API calls
3. Increase test coverage to 85%"""
    
    return [
        BaseTool.from_function(plan_project, description="Create project plan with phases, tasks, and estimates."),
        BaseTool.from_function(assess_risk, description="Assess project risks and provide mitigation strategies."),
        BaseTool.from_function(calculate, description="Perform calculations: basic, roi, npv, or estimate."),
        BaseTool.from_function(analyze_code, description="Analyze code for quality, complexity, and issues."),
    ]


# =============================================================================
# EXAMPLE 1: Research Agent - Information Gathering & Synthesis
# =============================================================================

async def example_research_agent():
    """
    Research Agent that:
    1. Searches multiple sources
    2. Analyzes findings
    3. Synthesizes into a report
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 1: Research Agent - Multi-Source Information Synthesis")
    logger.info("=" * 80)
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set!")
        return False
    
    llm = BaseOpenAI(model_name="gpt-4o-mini", temperature=0.3)
    tools = create_research_tools()
    
    # CHAIN_OF_THOUGHT: Encourages step-by-step reasoning
    prompt = PromptFactory.create_prompt(
        technique=PromptTechnique.CHAIN_OF_THOUGHT
    ).configure(
        system="""You are a senior research analyst specializing in technology trends.
Your task is to gather information from multiple sources, analyze patterns,
and synthesize findings into actionable insights.

When researching:
1. Search for relevant information
2. Query internal data for context
3. Analyze the combined data
4. Summarize findings
5. Generate a professional report""",
        user="Complete the research task using the available tools.",
        cot_instruction="Let's approach this research systematically, step by step, ensuring we gather and analyze all relevant data before drawing conclusions."
    )
    
    agent = Agent(
        name="ResearchAnalyst",
        role="Senior Research Analyst",
        objective="Gather, analyze, and synthesize information into actionable reports",
        llm=llm,
        prompt=prompt,
        tools=tools,
        config=AgentConfig(
            execution_mode=ExecutionMode.AUTONOMOUS,
            verbose=False,
            max_iterations=15,
            enable_memory=True,
            planning_max_tokens=2048,
            llm_max_tokens=2048
        )
    )
    
    await agent.initialize()
    
    task = Task(
        id="research_python_perf",
        objective="""Research Python performance optimization:
1. Search web for Python performance techniques
2. Query our metrics database for current baselines
3. Analyze the data for trends
4. Summarize findings in executive style
5. Generate a markdown report titled 'Python Performance Analysis'"""
    )
    
    logger.info(f"Task: {task.objective[:80]}...")
    logger.info("Mode: AUTONOMOUS")
    logger.info("-" * 80)
    
    result = await agent.execute(task)
    
    logger.info(f"\nResult:\n{result[:500]}..." if len(str(result)) > 500 else f"\nResult:\n{result}")
    logger.info(f"Agent State: {agent.state}")
    
    success = agent.state == AgentState.COMPLETED and "Error:" not in str(result)
    return success


# =============================================================================
# EXAMPLE 2: Data Analysis Pipeline Agent
# =============================================================================

async def example_data_pipeline_agent():
    """
    Data Pipeline Agent that:
    1. Queries multiple data sources
    2. Performs various analyses
    3. Generates insights and forecasts
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 2: Data Pipeline Agent - Complex Analytics Workflow")
    logger.info("=" * 80)
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set!")
        return False
    
    llm = BaseOpenAI(model_name="gpt-4o-mini", temperature=0.2)
    tools = create_research_tools()
    
    # FEW_SHOT: Learning from examples of data analysis
    prompt = PromptFactory.create_prompt(
        technique=PromptTechnique.FEW_SHOT
    ).configure(
        system="""You are a data analytics specialist.
Extract insights through systematic analysis:
1. Gather data from relevant sources
2. Perform exploratory analysis
3. Identify trends and patterns
4. Generate forecasts
5. Document findings""",
        user="Analyze the data step by step using the available tools.",
        examples=[
            {
                "input": "Analyze user growth for Q3",
                "output": "Step 1: Query users table. Step 2: Analyze trends. Step 3: Generate forecast. Result: 15% MoM growth with seasonal patterns."
            },
            {
                "input": "Create revenue report",
                "output": "Step 1: Query sales data. Step 2: Calculate key metrics. Step 3: Identify top products. Step 4: Generate markdown report with insights."
            }
        ]
    )
    
    agent = Agent(
        name="DataAnalyst",
        role="Data Analytics Specialist",
        objective="Extract insights through systematic data analysis and reporting",
        llm=llm,
        prompt=prompt,
        tools=tools,
        config=AgentConfig(
            execution_mode=ExecutionMode.AUTONOMOUS,
            verbose=False,
            max_iterations=15,
            enable_memory=True
        )
    )
    
    await agent.initialize()
    
    task = Task(
        id="q4_business_analysis",
        objective="""Analyze business performance and create a quarterly report:
1. Query 'users' table for user metrics
2. Query 'sales' table for revenue data
3. Perform trend analysis on the data
4. Generate a forecast for next quarter
5. Create a markdown report titled 'Q4 Business Analysis'"""
    )
    
    logger.info(f"Task: {task.objective[:80]}...")
    logger.info("Mode: AUTONOMOUS")
    logger.info("-" * 80)
    
    result = await agent.execute(task)
    
    logger.info(f"\nResult:\n{result[:500]}..." if len(str(result)) > 500 else f"\nResult:\n{result}")
    logger.info(f"Agent State: {agent.state}")
    
    success = agent.state == AgentState.COMPLETED and "Error:" not in str(result)
    return success


# =============================================================================
# EXAMPLE 3: Project Planning Agent
# =============================================================================

async def example_project_planning_agent():
    """
    Project Planning Agent that:
    1. Analyzes requirements
    2. Creates task breakdown
    3. Identifies risks
    4. Produces comprehensive plan
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 3: Project Planning Agent - Comprehensive Planning")
    logger.info("=" * 80)
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set!")
        return False
    
    llm = BaseOpenAI(model_name="gpt-4o-mini", temperature=0.3)
    tools = create_project_tools()
    
    # CHAIN_OF_THOUGHT: Logical step-by-step planning
    prompt = PromptFactory.create_prompt(
        technique=PromptTechnique.CHAIN_OF_THOUGHT
    ).configure(
        system="""You are a senior project manager with software development expertise.
Create comprehensive project plans accounting for:
- Technical complexity
- Team capacity
- Risk factors
- Dependencies
Provide realistic estimates and identify blockers.""",
        user="Create the project plan using the available tools.",
        cot_instruction="Let's think through this project systematically: first break down the work, then assess risks, calculate costs, and finally document everything."
    )
    
    agent = Agent(
        name="ProjectManager",
        role="Senior Project Manager",
        objective="Create comprehensive project plans with tasks, estimates, and risk assessments",
        llm=llm,
        prompt=prompt,
        tools=tools,
        config=AgentConfig(
            execution_mode=ExecutionMode.AUTONOMOUS,
            verbose=False,
            max_iterations=12,
            enable_memory=True
        )
    )
    
    await agent.initialize()
    
    task = Task(
        id="analytics_dashboard_plan",
        objective="""Create a project plan for building a customer analytics dashboard:
- Team: 4 developers
- Stack: React, Python, PostgreSQL

Required:
1. Break down project into phases and tasks (use plan_project with team_size=4)
2. Assess risks for this project
3. Calculate total cost assuming $150/hour rate (use calculate with calc_type='estimate')
4. Generate a markdown report titled 'Analytics Dashboard Plan'"""
    )
    
    logger.info(f"Task: {task.objective[:80]}...")
    logger.info("Mode: AUTONOMOUS")
    logger.info("-" * 80)
    
    result = await agent.execute(task)
    
    logger.info(f"\nResult:\n{result[:500]}..." if len(str(result)) > 500 else f"\nResult:\n{result}")
    logger.info(f"Agent State: {agent.state}")
    
    success = agent.state == AgentState.COMPLETED and "Error:" not in str(result)
    return success


# =============================================================================
# EXAMPLE 4: Code Quality Agent
# =============================================================================

async def example_code_quality_agent():
    """
    Code Quality Agent that:
    1. Analyzes code quality
    2. Identifies issues
    3. Calculates technical debt
    4. Provides improvement roadmap
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 4: Code Quality Agent - Technical Debt Analysis")
    logger.info("=" * 80)
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set!")
        return False
    
    llm = BaseOpenAI(model_name="gpt-4o-mini", temperature=0.2)
    tools = create_project_tools()
    
    # FEW_SHOT + CoT: Examples combined with step-by-step reasoning
    prompt = PromptFactory.create_prompt(
        technique=PromptTechnique.FEW_SHOT
    ).configure(
        system="""You are a software architect focused on code quality.
Your expertise includes:
- Static code analysis
- Technical debt assessment
- Refactoring strategies
Provide actionable recommendations prioritized by impact.""",
        user="Perform the code quality assessment using the available tools.",
        use_cot=True,
        cot_instruction="Let's analyze this codebase methodically: first scan for issues, then assess severity, calculate remediation effort, and create a prioritized action plan.",
        examples=[
            {
                "input": "Assess code quality for auth module",
                "output": "1. analyze_code('auth'): Found 5 issues. 2. assess_risk: Security vulnerabilities critical. 3. calculate cost: 40h remediation. 4. Recommendation: Prioritize security fixes."
            },
            {
                "input": "Review API endpoints",
                "output": "1. analyze_code('api'): Complexity=15. 2. assess_risk: Performance bottlenecks. 3. plan_project: 3 sprints needed. 4. Report: Refactor high-traffic endpoints first."
            }
        ]
    )
    
    agent = Agent(
        name="CodeQualityExpert",
        role="Software Architect",
        objective="Analyze code quality and provide actionable improvement recommendations",
        llm=llm,
        prompt=prompt,
        tools=tools,
        config=AgentConfig(
            execution_mode=ExecutionMode.AUTONOMOUS,
            verbose=False,
            max_iterations=12,
            enable_memory=True
        )
    )
    
    await agent.initialize()
    
    task = Task(
        id="code_quality_assessment",
        objective="""Perform code quality assessment for our backend service:
1. Analyze code at 'src/backend/main_service'
2. Assess technical risks based on findings
3. Calculate remediation cost at $100/hour (use calculate with calc_type='estimate')
4. Create a remediation plan (use plan_project with team_size=2)
5. Generate a markdown report titled 'Code Quality Report'"""
    )
    
    logger.info(f"Task: {task.objective[:80]}...")
    logger.info("Mode: AUTONOMOUS")
    logger.info("-" * 80)
    
    result = await agent.execute(task)
    
    logger.info(f"\nResult:\n{result[:500]}..." if len(str(result)) > 500 else f"\nResult:\n{result}")
    logger.info(f"Agent State: {agent.state}")
    
    success = agent.state == AgentState.COMPLETED and "Error:" not in str(result)
    return success


# =============================================================================
# EXAMPLE 5: Strategic Decision Agent
# =============================================================================

async def example_strategic_decision_agent():
    """
    Strategic Decision Agent that:
    1. Gathers market intelligence
    2. Analyzes capabilities
    3. Assesses risks
    4. Provides recommendations
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 5: Strategic Decision Agent - Market Analysis")
    logger.info("=" * 80)
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set!")
        return False
    
    llm = BaseOpenAI(model_name="gpt-4o-mini", temperature=0.4)
    
    # Combine research and project tools
    tools = create_research_tools() + create_project_tools()
    
    # CHAIN_OF_THOUGHT: Multi-phase strategic reasoning
    prompt = PromptFactory.create_prompt(
        technique=PromptTechnique.CHAIN_OF_THOUGHT
    ).configure(
        system="""You are a strategic advisor with technology market expertise.
Provide data-driven recommendations by:
1. Analyzing market trends
2. Evaluating resources
3. Assessing risks and opportunities
4. Quantifying outcomes
Support recommendations with data.""",
        user="Conduct the strategic analysis using the available tools.",
        cot_instruction="""Let's approach this strategic decision methodically:

Phase 1 - Market Intelligence: Gather external market data and trends.
Phase 2 - Internal Assessment: Query our current capabilities and metrics.
Phase 3 - Analysis: Synthesize findings and identify patterns.
Phase 4 - Risk Evaluation: Assess potential risks and mitigations.
Phase 5 - Financial Modeling: Calculate investments and projections.
Phase 6 - Synthesis: Summarize and formulate recommendation.
Phase 7 - Documentation: Generate comprehensive report.

Now, let's execute each phase systematically."""
    )
    
    agent = Agent(
        name="StrategicAdvisor",
        role="Strategic Advisor",
        objective="Provide data-driven strategic recommendations and market analysis",
        llm=llm,
        prompt=prompt,
        tools=tools,
        config=AgentConfig(
            execution_mode=ExecutionMode.AUTONOMOUS,
            verbose=False,
            max_iterations=18,
            enable_memory=True,
            planning_max_tokens=2048
        )
    )
    
    await agent.initialize()
    
    task = Task(
        id="ml_market_expansion",
        objective="""Evaluate ML market expansion opportunity:
1. Search for machine learning market trends
2. Search for startup funding trends in AI
3. Query our sales data for current revenue
4. Query user metrics to assess customer base
5. Analyze market data for trends
6. Assess risks of entering ML market
7. Calculate 6-month investment at $200K/month (use calculate with basic: 200000*6)
8. Summarize findings in executive style
9. Generate markdown report titled 'ML Market Expansion Analysis'

Provide GO/NO-GO recommendation with rationale."""
    )
    
    logger.info(f"Task: {task.objective[:80]}...")
    logger.info("Mode: AUTONOMOUS (complex multi-step analysis)")
    logger.info("-" * 80)
    
    result = await agent.execute(task)
    
    logger.info(f"\nResult:\n{result[:500]}..." if len(str(result)) > 500 else f"\nResult:\n{result}")
    logger.info(f"Agent State: {agent.state}")
    
    success = agent.state == AgentState.COMPLETED and "Error:" not in str(result)
    return success


# =============================================================================
# EXAMPLE 6: RAG-Enhanced Agent (Retrieval Augmented Generation)
# =============================================================================

async def example_rag_agent():
    """
    RAG Agent that:
    1. Uses retrieved context for grounded responses
    2. Combines external knowledge with tool execution
    3. Produces fact-based analysis
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 6: RAG Agent - Retrieval Augmented Generation")
    logger.info("=" * 80)
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set!")
        return False
    
    llm = BaseOpenAI(model_name="gpt-4o-mini", temperature=0.2)
    tools = create_research_tools()
    
    # Simulated retrieved context (in real RAG, this comes from vector DB)
    retrieved_context = """
RETRIEVED KNOWLEDGE BASE CONTEXT:

Document 1: Python Performance Best Practices (2024)
- Use Python 3.12+ for 25% performance improvement
- Profile before optimizing (use cProfile, py-spy)
- Consider Cython for compute-intensive loops
- Use multiprocessing for CPU-bound tasks, asyncio for I/O-bound

Document 2: Company Performance Metrics (Internal)
- Current Python version: 3.11.5
- Average API response time: 245ms
- Database query latency: 120ms
- Memory usage: 2.1GB average per worker

Document 3: Industry Benchmarks
- Top quartile API response: <100ms
- Optimal memory per worker: <1GB
- Modern Python adoption rate: 78% on 3.11+
"""
    
    # RETRIEVAL_AUGMENTED_GENERATION: Grounded in retrieved knowledge
    prompt = PromptFactory.create_prompt(
        technique=PromptTechnique.RETRIEVAL_AUGMENTED_GENERATION
    ).configure(
        system="""You are an expert technical advisor. Use the retrieved context 
to provide accurate, fact-based recommendations. Always ground your 
analysis in the provided knowledge base documents. Cross-reference 
internal metrics with industry benchmarks.""",
        context=retrieved_context,
        user="Analyze the retrieved information and use tools to provide additional insights."
    )
    
    agent = Agent(
        name="RAGAdvisor",
        role="Technical Advisor with Knowledge Base",
        objective="Provide grounded technical recommendations using retrieved context",
        llm=llm,
        prompt=prompt,
        tools=tools,
        config=AgentConfig(
            execution_mode=ExecutionMode.AUTONOMOUS,
            verbose=False,
            max_iterations=12,
            enable_memory=True
        )
    )
    
    await agent.initialize()
    
    task = Task(
        id="rag_performance_analysis",
        objective="""Based on the retrieved context about Python performance and our internal metrics:
1. Search for latest Python 3.12 performance improvements
2. Query our metrics database for current performance data
3. Analyze the gap between our metrics and industry benchmarks
4. Summarize findings with specific upgrade recommendations
5. Generate a markdown report titled 'Performance Optimization Plan'

Ground all recommendations in the retrieved knowledge base context."""
    )
    
    logger.info(f"Task: {task.objective[:80]}...")
    logger.info("Mode: AUTONOMOUS with RAG prompting")
    logger.info("Technique: RETRIEVAL_AUGMENTED_GENERATION")
    logger.info("-" * 80)
    
    result = await agent.execute(task)
    
    logger.info(f"\nResult:\n{result[:500]}..." if len(str(result)) > 500 else f"\nResult:\n{result}")
    logger.info(f"Agent State: {agent.state}")
    
    success = agent.state == AgentState.COMPLETED and "Error:" not in str(result)
    return success


# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Run all complex autonomous agent examples with various prompt techniques."""
    logger.info("=" * 80)
    logger.info("NucleusIQ Complex Autonomous Agent Examples")
    logger.info("=" * 80)
    logger.info("""
This suite demonstrates AUTONOMOUS mode with MULTIPLE PROMPT TECHNIQUES:

1. Research Agent         - CHAIN_OF_THOUGHT   (step-by-step reasoning)
2. Data Pipeline Agent    - FEW_SHOT           (learning from examples)
3. Project Planning Agent - CHAIN_OF_THOUGHT   (logical planning)
4. Code Quality Agent     - FEW_SHOT + CoT     (examples + reasoning)
5. Strategic Decision     - CHAIN_OF_THOUGHT   (multi-phase reasoning)
6. RAG Agent             - RAG                 (retrieval augmented)

Prompt Techniques Demonstrated:
- CHAIN_OF_THOUGHT: Explicit step-by-step reasoning chains
- FEW_SHOT: In-context learning from input/output examples
- FEW_SHOT + CoT: Hybrid combining examples with reasoning
- RAG: Grounding responses in retrieved knowledge

Each example also showcases:
- Automatic multi-step planning
- Tool orchestration  
- Context building across steps
""")
    logger.info("=" * 80)
    
    results = {}
    
    # Run examples with different prompt techniques
    examples = [
        ("Research Agent (CoT)", example_research_agent),
        ("Data Pipeline (FewShot)", example_data_pipeline_agent),
        ("Project Planning (CoT)", example_project_planning_agent),
        ("Code Quality (FewShot+CoT)", example_code_quality_agent),
        ("Strategic Decision (CoT)", example_strategic_decision_agent),
        ("RAG Agent", example_rag_agent),
    ]
    
    for name, example_fn in examples:
        try:
            logger.info(f"\nRunning: {name}")
            success = await example_fn()
            results[name] = success
            logger.info(f"{name}: {'PASS' if success else 'FAIL'}")
        except Exception as e:
            logger.error(f"{name} failed: {e}")
            results[name] = False
        
        logger.info("-" * 80)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, success in results.items():
        status = "PASS" if success else "FAIL"
        logger.info(f"  {name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} examples completed successfully")
    
    if passed == total:
        logger.info("\nAll prompt techniques executed successfully!")
        logger.info("Demonstrated: CHAIN_OF_THOUGHT, FEW_SHOT, FEW_SHOT+CoT, RAG")
    else:
        logger.warning(f"\n{total - passed} example(s) had issues.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
