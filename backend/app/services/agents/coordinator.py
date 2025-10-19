"""
app/agents/coordinator.py - Multi-Agent Orchestration
"""
import logging
from typing import Dict, Any, List
from datetime import datetime

from app.services.agents.base_agent import BaseAgent
from app.services.agents.researcher_agent import ResearcherAgent
from app.services.agents.writer_agent import WriterAgent
from app.services.agents.critic_agent import CriticAgent
from app.core.enums import AgentType, AgentStatus

logger = logging.getLogger(__name__)


class CoordinatorAgent:
    """
    Coordinator Agent: Orchestrates multi-agent collaboration
    
    Workflow:
    1. Researcher gathers information
    2. Writer creates content
    3. Critic evaluates quality
    4. Iterate if needed
    5. Return final answer
    """
    
    def __init__(
        self,
        researcher: ResearcherAgent,
        writer: WriterAgent,
        critic: CriticAgent,
        max_iterations: int = 3
    ):
        self.researcher = researcher
        self.writer = writer
        self.critic = critic
        self.max_iterations = max_iterations
        
        self.execution_log: List[Dict[str, Any]] = []
        
        logger.info("Coordinator initialized with 3 agents")
    
    async def execute(
        self,
        query: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Orchestrate multi-agent collaboration
        
        Args:
            query: User query
            context: Additional context
            
        Returns:
            Final answer with collaboration details
        """
        start_time = datetime.now()
        context = context or {}
        
        logger.info(f"Starting multi-agent collaboration for: {query}")
        
        iteration = 0
        final_answer = None
        all_steps = []
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Collaboration iteration {iteration}/{self.max_iterations}")
            
            try:
                # Phase 1: Research
                research_result = await self._research_phase(query, context)
                all_steps.extend(research_result.get("steps", []))
                
                if research_result["status"] != "success":
                    logger.error("Research phase failed")
                    break
                
                # Phase 2: Write
                write_result = await self._write_phase(query, research_result)
                all_steps.extend(write_result.get("steps", []))
                
                if write_result["status"] != "success":
                    logger.error("Write phase failed")
                    break
                
                # Phase 3: Critique
                critique_result = await self._critique_phase(
                    query, research_result, write_result
                )
                all_steps.extend(critique_result.get("steps", []))
                
                # Check if approved
                if critique_result["verdict"] == "APPROVED":
                    final_answer = write_result["output"]
                    logger.info("Content approved by critic")
                    break
                
                # If not approved, provide feedback for next iteration
                context["critique_feedback"] = critique_result["issues"]
                logger.info(f"Revision needed. Issues: {len(critique_result['issues'])}")
                
            except Exception as e:
                logger.error(f"Collaboration error: {e}")
                break
        
        # Generate final response
        if final_answer is None:
            final_answer = await self._generate_fallback_answer(query, all_steps)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Compile collaboration summary
        collaboration_summary = self._create_collaboration_summary(
            iteration, all_steps
        )
        
        return {
            "final_answer": final_answer,
            "collaboration_summary": collaboration_summary,
            "total_iterations": iteration,
            "agent_executions": [
                self._format_agent_execution(self.researcher),
                self._format_agent_execution(self.writer),
                self._format_agent_execution(self.critic)
            ],
            "all_steps": all_steps,
            "processing_time": execution_time,
            "status": "success" if final_answer else "failed"
        }
    
    async def _research_phase(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute research phase"""
        logger.info("Phase 1: Research")
        
        try:
            result = await self.researcher.execute(query, context)
            
            self.execution_log.append({
                "phase": "research",
                "agent": "researcher",
                "status": result["status"],
                "confidence": result.get("confidence", 0.0)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Research phase error: {e}")
            return {
                "output": "",
                "status": "failed",
                "confidence": 0.0,
                "steps": []
            }
    
    async def _write_phase(
        self,
        query: str,
        research_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute writing phase"""
        logger.info("Phase 2: Writing")
        
        try:
            write_context = {
                "research_findings": research_result["output"],
                "sources": research_result.get("sources", [])
            }
            
            result = await self.writer.execute(query, write_context)
            
            self.execution_log.append({
                "phase": "writing",
                "agent": "writer",
                "status": result["status"],
                "confidence": result.get("confidence", 0.0)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Writing phase error: {e}")
            return {
                "output": "",
                "status": "failed",
                "confidence": 0.0,
                "steps": []
            }
    
    async def _critique_phase(
        self,
        query: str,
        research_result: Dict[str, Any],
        write_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute critique phase"""
        logger.info("Phase 3: Critique")
        
        try:
            critique_context = {
                "content": write_result["output"],
                "research_findings": research_result["output"]
            }
            
            result = await self.critic.execute(query, critique_context)
            
            self.execution_log.append({
                "phase": "critique",
                "agent": "critic",
                "status": result["status"],
                "score": result.get("score", 0.0),
                "verdict": result.get("verdict", "UNKNOWN")
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Critique phase error: {e}")
            return {
                "output": "",
                "status": "failed",
                "score": 0.0,
                "verdict": "FAILED",
                "issues": [],
                "steps": []
            }
    
    async def _generate_fallback_answer(
        self,
        query: str,
        all_steps: List
    ) -> str:
        """Generate fallback answer if collaboration fails"""
        # Try to extract best available content
        for log_entry in reversed(self.execution_log):
            if log_entry["phase"] == "writing" and log_entry["status"] == "success":
                # Return the last successful write result
                return self.writer.execution_history[-1].observation if self.writer.execution_history else \
                    "I apologize, but I was unable to generate a complete answer."
        
        return "I apologize, but the multi-agent system was unable to generate a satisfactory answer. Please try rephrasing your question."
    
    def _create_collaboration_summary(
        self,
        iterations: int,
        all_steps: List
    ) -> str:
        """Create human-readable collaboration summary"""
        summary = f"Multi-Agent Collaboration Summary:\n"
        summary += f"{'='*50}\n"
        summary += f"Total Iterations: {iterations}\n"
        summary += f"Total Steps: {len(all_steps)}\n\n"
        
        for log_entry in self.execution_log:
            phase = log_entry["phase"].title()
            agent = log_entry["agent"].title()
            status = log_entry["status"].upper()
            
            summary += f"{phase} ({agent}): {status}\n"
            
            if "confidence" in log_entry:
                summary += f"  Confidence: {log_entry['confidence']:.2f}\n"
            if "score" in log_entry:
                summary += f"  Quality Score: {log_entry['score']}/10\n"
            if "verdict" in log_entry:
                summary += f"  Verdict: {log_entry['verdict']}\n"
            
            summary += "\n"
        
        return summary
    
    def _format_agent_execution(self, agent: BaseAgent) -> Dict[str, Any]:
        """Format agent execution details"""
        return {
            "agent_type": agent.agent_type.value,
            "status": agent.status.value,
            "total_steps": len(agent.execution_history),
            "tools_used": len(agent.tools)
        }
    
    def reset(self):
        """Reset all agents for new execution"""
        self.researcher.clear_history()
        self.writer.clear_history()
        self.critic.clear_history()
        self.execution_log.clear()
        logger.info("Coordinator reset - ready for new task")