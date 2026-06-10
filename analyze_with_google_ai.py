
import asyncio
import json
from pathlib import Path

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.prompts.zero_shot import ZeroShotPrompt
from nucleusiq_gemini import BaseGemini
import os

HERE = Path(__file__).resolve().parent
CHALLENGE_DIR = HERE / "notebooks/agents/agent_engineering_challenge/"
DATA_DIR = CHALLENGE_DIR / "data"


async def main() -> None:
    # Read the data files
    file_contents = []
    for file_path in DATA_DIR.glob("*.txt"):
        file_contents.append(f"--- {file_path.name} ---\n{file_path.read_text()}\n")

    # The prompt for the model
    prompt = f'''
    You are a careful investment-risk analyst. Analyze the following internal documents for Aurora Retail Systems.
    Produce a single response containing all four sections as described in the task.

    Here are the documents:
    {"".join(file_contents)}

    Final Recommendation: <one paragraph>

    Top 5 Risks:
      1. <risk title> — <severity>
         Evidence:
           - <quoted or paraphrased finding> (<source file>)
           - ...
      2. ...

    Unknowns / Diligence Questions:
      - <question>
      - ...
    '''

    agent = Agent(
        name="analyst",
        prompt=ZeroShotPrompt().configure(
            system="You are a careful investment-risk analyst.",
        ),
        llm=BaseGemini(model_name="gemini-2.5-flash"),
        config=AgentConfig(execution_mode=ExecutionMode.AUTONOMOUS),
    )

    await agent.initialize()
    result = await agent.execute(
        Task(id="analysis-1", objective=prompt),
    )

    # Create a simplified scorecard
    scorecard = {
        "model": "gemini-2.5-flash",
        "provider": "gemini",
        "final_answer": result.output,
        "notes": "Using NucleusIQ framework with Gemini provider. Telemetry data was not available in the result object.",
    }

    # Save the result
    with open("result.json", "w") as f:
        json.dump(scorecard, f, indent=2)

    print("Analysis complete. result.json has been generated.")


if __name__ == "__main__":
    asyncio.run(main())
