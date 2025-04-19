# src/nucleusiq/agents/agent.py
from typing import Any, Dict, List, Optional
from datetime import datetime

from nucleusiq.agents.builder.base_anget import BaseAgent
from nucleusiq.agents.config.agent_config import AgentState, AgentMetrics
# from nucleusiq.core.memory import BaseMemory
from nucleusiq.prompts.base import BasePrompt
from nucleusiq.llms.base_llm import BaseLLM
# from nucleusiq.core.tools import BaseTool

class Agent(BaseAgent):
    """
    Concrete implementation of an agent in the NucleusIQ framework.
    
    This class provides the actual implementation of agent behaviors,
    building upon the foundation established in BaseAgent.
    """

    async def initialize(self) -> None:
        """Initialize agent components and resources."""
        self._logger.info(f"Initializing agent: {self.name}")
        
        try:
            # Initialize memory if provided
            if self.memory:
                await self.memory.initialize()
                self._logger.debug("Memory system initialized")
            
            # Initialize prompt if provided
            if self.prompt:
                prompt_text = self.prompt.format_prompt()
                self._logger.debug(f"Prompt system initialized \n {prompt_text}")
            
            # Initialize tools
            # for tool in self.tools:
            #     await tool.initialize()
            # self._logger.debug(f"Initialized {len(self.tools)} tools")
            
            # Mark initialization as complete
            self.state = AgentState.COMPLETED
            self._logger.info("Agent initialization completed successfully")
            
        except Exception as e:
            self.state = AgentState.ERROR
            self._logger.error(f"Agent initialization failed: {str(e)}")
            raise

    # Placeholder for other methods like execute, plan, etc.
    # These will be implemented in subsequent steps. 
            plan = await self._create_basic_plan(task)
            
            return plan
            
        except Exception as e:
            self._logger.error(f"Planning failed: {str(e)}")
            raise

    async def _get_context(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant context for task execution."""
        context = {
            "task": task,
            "agent_role": self.role,
            "agent_objective": self.objective,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add memory context if available
        if self.memory:
            memory_context = await self.memory.get_relevant_context(task)
            context["memory"] = memory_context
            
        return context

    async def _create_llm_plan(
        self,
        task: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create an execution plan using the LLM."""
        # Construct planning prompt
        plan_prompt = self._construct_planning_prompt(task, context)
        
        # Generate plan using LLM
        try:
            plan_response = await self.llm.generate(plan_prompt)
            return self._parse_plan_response(plan_response)
        except Exception as e:
            self._logger.error(f"LLM planning failed: {str(e)}")
            return await self._create_basic_plan(task)

    async def _create_basic_plan(
        self,
        task: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create a basic execution plan without LLM."""
        # Simple single-step plan
        return [{
            "step": 1,
            "action": "execute",
            "task": task
        }]

    def _construct_planning_prompt(
        self,
        task: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Construct a prompt for plan generation."""
        if self.prompt:
            return self.prompt.format(
                task=task,
                context=context,
                tools=self.tools,
                agent_role=self.role
            )
        
        # Basic prompt construction
        return f"""
        As {self.role} with objective '{self.objective}',
        create a plan to accomplish the following task:
        {task}
        
        Available tools: {[t.name for t in self.tools]}
        
        Create a step-by-step plan to accomplish this task.
        """

    def _parse_plan_response(
        self,
        response: str
    ) -> List[Dict[str, Any]]:
        """Parse the LLM's planning response into structured steps."""
        # This is a simplified version - in practice, you'd want more sophisticated parsing
        steps = []
        current_step = None
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Step'):
                if current_step:
                    steps.append(current_step)
                current_step = {
                    'step': len(steps) + 1,
                    'action': '',
                    'details': ''
                }
            elif current_step:
                if not current_step['action']:
                    current_step['action'] = line
                else:
                    current_step['details'] += line + '\n'
                    
        if current_step:
            steps.append(current_step)
            
        return steps

    async def _process_result(self, result: Any) -> Any:
        """Process and store execution results."""
        try:
            # Store in memory if available
            if self.memory:
                await self.memory.store(
                    key=f"task_result_{datetime.now().isoformat()}",
                    value={
                        'task': self._current_task,
                        'result': result,
                        'metrics': self.metrics.dict()
                    }
                )
            
            # Process through prompt if available
            if self.prompt:
                result = await self.prompt.process_result(result)
            
            return result
            
        except Exception as e:
            self._logger.error(f"Result processing failed: {str(e)}")
            raise

    def _validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate task format and requirements."""
        required_fields = ['id', 'objective']
        return all(field in task for field in required_fields)

    async def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute a specific tool with parameters."""
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
            
        self.state = AgentState.WAITING_FOR_TOOLS
        try:
            return await tool.execute(**params)
        finally:
            self.state = AgentState.EXECUTING

    async def _handle_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> None:
        """Handle execution errors with appropriate logging and recovery."""
        self._logger.error(f"Error during execution: {str(error)}")
        
        if self.memory:
            await self.memory.store(
                key=f"error_{datetime.now().isoformat()}",
                value={
                    'error': str(error),
                    'context': context,
                    'task': self._current_task
                }
            )
            
        self.metrics.error_count += 1
        self.state = AgentState.ERROR

    async def save_state(self) -> Dict[str, Any]:
        """Save agent's current state."""
        state = {
            'id': self.id,
            'name': self.name,
            'state': self.state,
            'metrics': self.metrics.dict(),
            'current_task': self._current_task,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.memory:
            state['memory'] = await self.memory.export_state()
            
        return state

    async def load_state(self, state: Dict[str, Any]) -> None:
        """Load agent's saved state."""
        self.state = state['state']
        self.metrics = AgentMetrics(**state['metrics'])
        self._current_task = state['current_task']
        
        if self.memory and 'memory' in state:
            await self.memory.import_state(state['memory'])
            
        self._logger.info(f"Loaded agent state from {state['timestamp']}")

    async def delegate_task(
        self,
        task: Dict[str, Any],
        target_agent: 'Agent'
    ) -> Any:
        """Delegate a task to another agent."""
        self._logger.info(f"Delegating task to agent to perfoming the task: {target_agent.name}")
        self.state = AgentState.WAITING_FOR_HUMAN
        
        try:
            return await target_agent.execute(task)
        finally:
            self.state = AgentState.EXECUTING