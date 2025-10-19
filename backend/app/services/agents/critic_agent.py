"""
=====================================================================
5. app/services/agents/critic_agent.py - Critic Agent
=====================================================================
"""
from typing import Dict, Any
import logging
import re

from app.services.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """
    Critic Agent: Specialized in quality evaluation and review
    
    Responsibilities:
    - Evaluate content quality
    - Check factual accuracy
    - Assess completeness
    - Suggest improvements
    - Assign quality scores
    """
    
    def __init__(self, llm_service, max_iterations: int = 1):
        """
        Initialize Critic Agent
        
        Args:
            llm_service: LLM service for generation
            max_iterations: Max evaluation iterations
        """
        super().__init__(
            agent_type="critic",
            llm_service=llm_service,
            max_iterations=max_iterations
        )
    
    def get_system_prompt(self) -> str:
        """Get critic system prompt"""
        return """You are an expert Critic Agent. Your role is to:

1. EVALUATE content quality objectively
2. CHECK factual accuracy against sources
3. ASSESS logical consistency
4. IDENTIFY gaps or weaknesses
5. SUGGEST specific improvements

Evaluation Criteria:
- Accuracy: Does it match the research?
- Completeness: Are all aspects covered?
- Clarity: Is it easy to understand?
- Relevance: Does it answer the question?
- Sources: Are sources properly used?

Output Format:
## Quality Assessment

### Score: [X/10]

### Strengths:
- [Strength 1]
- [Strength 2]

### Issues:
- [Issue 1]
- [Issue 2]

### Recommendations:
- [Recommendation 1]
- [Recommendation 2]

### Verdict: [APPROVED / NEEDS REVISION]
"""
    
    async def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute critique task
        
        Args:
            task: Original question
            context: Content to evaluate and research
            
        Returns:
            Quality assessment with score and verdict
        """
        context = context or {}
        
        logger.info(f"Critic evaluating content")
        
        try:
            content = context.get("content", "")
            research = context.get("research_findings", "")
            
            # Evaluate content
            evaluation = await self._evaluate_content(task, content, research)
            
            # Parse evaluation results
            score, verdict, issues = self._parse_evaluation(evaluation)
            
            self.add_to_history({
                "action": "evaluation_completed",
                "score": score,
                "verdict": verdict
            })
            
            return {
                "status": "success",
                "output": evaluation,
                "score": score,
                "verdict": verdict,
                "issues": issues,
                "confidence": 0.90
            }
            
        except Exception as e:
            logger.error(f"Critic error: {e}")
            return {
                "status": "failed",
                "output": f"Evaluation failed: {str(e)}",
                "score": 0,
                "verdict": "FAILED",
                "issues": [],
                "confidence": 0.0
            }
    
    async def _evaluate_content(
        self,
        question: str,
        content: str,
        research: str
    ) -> str:
        """Perform content evaluation"""
        
        prompt = f"""{self.get_system_prompt()}

Original Question:
{question}

Research Findings:
{research[:1000]}...

Content to Evaluate:
{content}

Provide detailed evaluation:"""
        
        evaluation = await self.generate_response(prompt, temperature=0.3)
        
        self.add_to_history({
            "action": "evaluate",
            "question": question
        })
        
        return evaluation
    
    def _parse_evaluation(self, evaluation: str) -> tuple:
        """Parse evaluation to extract score, verdict, and issues"""
        
        # Extract score (X/10)
        score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', evaluation)
        score = float(score_match.group(1)) if score_match else 5.0
        
        # Extract verdict
        verdict = "APPROVED" if "APPROVED" in evaluation else "NEEDS REVISION"
        
        # Extract issues
        issues = []
        issues_section = re.search(r'Issues?:(.*?)(?=###|$)', evaluation, re.DOTALL)
        if issues_section:
            issue_lines = issues_section.group(1).strip().split('\n')
            issues = [line.strip('- ').strip() for line in issue_lines if line.strip() and line.strip().startswith('-')]
        
        return score, verdict, issues[:5]  # Limit to 5 issues