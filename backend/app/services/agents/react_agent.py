"""
app/agents/react_agent.py - ReAct Pattern Implementation
Reasoning + Acting paradigm for step-by-step problem solving
"""
import re
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from app.agents.base_agent import BaseAgent
from app.core.enums import AgentType, AgentStatus
from app.core.config import settings

logger = logging.getLogger(__name__)


class ReActAgent(BaseAgent):
    """
    ReAct Agent: Combines Reasoning and Acting
    
    Workflow:
    1. Thought: Reason about what to do next
    2. Action: Choose a tool and inputs
    3. Observation: Observe tool output
    4. Repeat until Final Answer
    """
    
    def __init__(self, llm_service, tools, max_iterations: int = None):
        super().__init__(
            agent_type=AgentType.REACT,
            llm_service=llm_service,
            tools=tools,
            max_iterations=max_iterations or settings.MAX_REACT_STEPS
        )
    
    def get_system_prompt(self) -> str:
        """ReAct system prompt with reasoning format"""
        return f"""You are a ReAct (Reasoning + Acting) agent that solves problems step by step.

You have access to the following tools:
{self.get_available_tools_description()}

Use the following format STRICTLY:

Thought: [Your reasoning about what to do next]
Action: [Tool name to use]
Action Input: {{"param1": "value1", "param2": "value2"}}
Observation: [Tool result - will be provided]

... (repeat Thought/Action/Observation as needed)

Thought: [Final reasoning]
Final Answer: [Complete answer to the user's question]

RULES:
1. Always start with "Thought:"
2. Choose ONE action at a time
3. Wait for Observation before next step
4. Use "Final Answer:" when you have enough information
5. Be concise and specific in your thoughts
6. Action Input must be valid JSON

Begin!"""
    
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute ReAct loop
        
        Args:
            task: User query/task
            context: Additional context (retrieved docs, etc.)
            
        Returns:
            Execution result with answer and steps
        """
        self.clear_history()
        self.update_status(AgentStatus.THINKING)
        
        start_time = datetime.now()
        
        # Build initial prompt
        context_str = self._format_context(context)
        prompt = f"""{self.get_system_prompt()}

Context Information:
{context_str}

Question: {task}

Thought:"""
        
        # ReAct loop
        iteration = 0
        agent_scratchpad = ""
        final_answer = None
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"ReAct iteration {iteration}/{self.max_iterations}")
            
            try:
                # Generate next thought and action
                full_prompt = prompt + agent_scratchpad
                response = await self.think(full_prompt)
                
                # Parse response
                parsed = self._parse_response(response)
                
                if parsed["type"] == "final_answer":
                    final_answer = parsed["content"]
                    logger.info("ReAct reached final answer")
                    break
                
                elif parsed["type"] == "action":
                    thought = parsed["thought"]
                    action = parsed["action"]
                    action_input = parsed["action_input"]
                    
                    # Execute action
                    observation = await self.act(action, action_input)
                    
                    # Record step
                    self.add_step(
                        thought=thought,
                        action=action,
                        action_input=action_input,
                        observation=observation
                    )
                    
                    # Update scratchpad
                    agent_scratchpad += f"""
Thought: {thought}
Action: {action}
Action Input: {action_input}
Observation: {observation}

"""
                else:
                    logger.warning(f"Could not parse response: {response}")
                    break
                    
            except Exception as e:
                logger.error(f"ReAct iteration error: {e}")
                final_answer = f"Error during execution: {str(e)}"
                break
        
        # Handle max iterations
        if final_answer is None:
            final_answer = await self._generate_fallback_answer(task, agent_scratchpad)
        
        self.update_status(AgentStatus.COMPLETED)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "answer": final_answer,
            "steps": self.execution_history,
            "iterations": iteration,
            "execution_time": execution_time,
            "status": "success" if final_answer else "failed"
        }
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for prompt"""
        if not context:
            return "No additional context provided."
        
        formatted = []
        
        if "retrieved_docs" in context:
            formatted.append("Retrieved Documents:")
            for i, doc in enumerate(context["retrieved_docs"][:3], 1):
                formatted.append(f"{i}. {doc['content'][:300]}...")
        
        if "graph_entities" in context:
            formatted.append(f"\nRelated Entities: {', '.join(context['graph_entities'][:5])}")
        
        if "memory" in context:
            formatted.append(f"\nConversation History: {context['memory']}")
        
        return "\n".join(formatted)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response into thought/action/observation format
        
        Returns:
            Dict with type ("action" or "final_answer") and parsed content
        """
        # Check for final answer
        if "Final Answer:" in response:
            match = re.search(r"Final Answer:\s*(.+)", response, re.DOTALL)
            if match:
                return {
                    "type": "final_answer",
                    "content": match.group(1).strip()
                }
        
        # Parse thought, action, and action input
        thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\n\n|$)", response, re.DOTALL)
        action_match = re.search(r"Action:\s*(\w+)", response)
        action_input_match = re.search(r"Action Input:\s*(\{.+?\})", response, re.DOTALL)
        
        if thought_match and action_match:
            thought = thought_match.group(1).strip()
            action = action_match.group(1).strip()
            
            # Parse action input JSON
            action_input = {}
            if action_input_match:
                try:
                    import json
                    action_input = json.loads(action_input_match.group(1))
                except:
                    # Fallback: extract key-value pairs
                    action_input = {"query": thought}
            
            return {
                "type": "action",
                "thought": thought,
                "action": action,
                "action_input": action_input
            }
        
        return {"type": "unknown", "content": response}
    
    async def _generate_fallback_answer(self, task: str, scratchpad: str) -> str:
        """Generate answer when max iterations reached"""
        prompt = f"""Based on the following reasoning process, provide a final answer to the question.

Question: {task}

Reasoning Process:
{scratchpad}

Provide a concise final answer:"""
        
        try:
            answer = await self.think(prompt)
            return answer
        except:
            return "I apologize, but I was unable to complete the reasoning process. Please try rephrasing your question."