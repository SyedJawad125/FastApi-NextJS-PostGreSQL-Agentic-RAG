"""
=====================================================================
2. app/services/agents/base_agent.py - Base Agent Class
=====================================================================
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    Provides common functionality for agent operations.
    """
    
    def __init__(self, agent_type: str, llm_service, max_iterations: int = 3):
        """
        Initialize base agent
        
        Args:
            agent_type: Type of agent (researcher, writer, critic)
            llm_service: LLM service instance for generation
            max_iterations: Maximum iterations for agent execution
        """
        self.agent_type = agent_type
        self.llm_service = llm_service
        self.max_iterations = max_iterations
        self.execution_history: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized {agent_type} agent")
    
    @abstractmethod
    async def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute agent task. Must be implemented by subclasses.
        
        Args:
            task: Task description
            context: Additional context for execution
            
        Returns:
            Dict containing execution results
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get agent-specific system prompt"""
        pass
    
    def add_to_history(self, step: Dict[str, Any]):
        """Add execution step to history"""
        step["timestamp"] = datetime.now().isoformat()
        self.execution_history.append(step)
    
    def clear_history(self):
        """Clear execution history"""
        self.execution_history.clear()
    
    def get_history_summary(self) -> str:
        """Get formatted execution history"""
        if not self.execution_history:
            return "No execution history."
        
        summary = []
        for i, step in enumerate(self.execution_history, 1):
            summary.append(f"Step {i}: {step.get('action', 'N/A')}")
        
        return "\n".join(summary)
    
    async def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate response using LLM
        
        Args:
            prompt: Input prompt
            temperature: Temperature for generation
            
        Returns:
            Generated response text
        """
        try:
            response = await self.llm_service.generate(
                prompt=prompt,
                temperature=temperature
            )
            return response
        except Exception as e:
            logger.error(f"Generation error in {self.agent_type}: {e}")
            return f"Error generating response: {str(e)}"
