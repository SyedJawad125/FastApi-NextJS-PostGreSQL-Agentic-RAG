"""
=====================================================================
4. app/services/agents/writer_agent.py - Writer Agent
=====================================================================
"""
from typing import Dict, Any, List
import logging

from app.services.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """
    Writer Agent: Specialized in content generation and writing
    
    Responsibilities:
    - Transform research into clear content
    - Structure information logically
    - Adapt tone and style
    - Ensure readability
    """
    
    def __init__(self, llm_service, max_iterations: int = 2):
        """
        Initialize Writer Agent
        
        Args:
            llm_service: LLM service for generation
            max_iterations: Max writing iterations
        """
        super().__init__(
            agent_type="writer",
            llm_service=llm_service,
            max_iterations=max_iterations
        )
    
    def get_system_prompt(self) -> str:
        """Get writer system prompt"""
        return """You are an expert Writer Agent. Your role is to:

1. TRANSFORM research findings into clear, engaging content
2. STRUCTURE information logically
3. ADAPT tone to the audience
4. ENSURE clarity and readability
5. MAINTAIN factual accuracy

Writing Principles:
- Start with a clear, direct answer
- Support with evidence and examples
- Use logical flow and transitions
- Be concise but comprehensive
- Cite sources when relevant
- Use appropriate formatting

Output Format:
[Well-structured, clear answer that directly addresses the question]
"""
    
    async def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute writing task
        
        Args:
            task: Original question/topic
            context: Research findings and sources
            
        Returns:
            Written content
        """
        context = context or {}
        
        logger.info(f"Writer starting task: {task}")
        
        try:
            research_findings = context.get("research_findings", "")
            sources = context.get("sources", [])
            
            # Generate initial content
            content = await self._generate_content(task, research_findings, sources)
            
            # Improve readability
            improved_content = await self._improve_readability(content)
            
            self.add_to_history({
                "action": "writing_completed",
                "task": task,
                "content_length": len(improved_content)
            })
            
            return {
                "status": "success",
                "output": improved_content,
                "confidence": 0.88
            }
            
        except Exception as e:
            logger.error(f"Writer error: {e}")
            return {
                "status": "failed",
                "output": f"Writing failed: {str(e)}",
                "confidence": 0.0
            }
    
    async def _generate_content(
        self,
        question: str,
        research: str,
        sources: List[str]
    ) -> str:
        """Generate initial content from research"""
        
        sources_text = "\n".join([f"- {s}" for s in sources]) if sources else "No sources"
        
        prompt = f"""{self.get_system_prompt()}

Question: {question}

Research Findings:
{research}

Sources:
{sources_text}

Write a clear, well-structured answer:"""
        
        content = await self.generate_response(prompt, temperature=0.7)
        
        self.add_to_history({
            "action": "generate_content",
            "question": question
        })
        
        return content
    
    async def _improve_readability(self, content: str) -> str:
        """Improve content readability"""
        
        prompt = f"""Improve the readability and flow of this content:

{content}

Make it:
1. More concise where possible
2. Better structured
3. Easier to understand
4. More engaging

Improved version:"""
        
        improved = await self.generate_response(prompt, temperature=0.5)
        
        self.add_to_history({
            "action": "improve_readability"
        })
        
        return improved
