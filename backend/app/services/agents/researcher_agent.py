"""
=====================================================================
3. app/services/agents/researcher_agent.py - Researcher Agent
=====================================================================
"""
from typing import Dict, Any, List
import logging

from app.services.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """
    Researcher Agent: Specialized in information gathering and research
    
    Responsibilities:
    - Search for relevant information
    - Extract key facts and data
    - Identify credible sources
    - Organize research findings
    """
    
    def __init__(self, llm_service, vectorstore_service=None, max_iterations: int = 3):
        """
        Initialize Researcher Agent
        
        Args:
            llm_service: LLM service for generation
            vectorstore_service: Vector store for document search
            max_iterations: Max research iterations
        """
        super().__init__(
            agent_type="researcher",
            llm_service=llm_service,
            max_iterations=max_iterations
        )
        self.vectorstore = vectorstore_service
    
    def get_system_prompt(self) -> str:
        """Get researcher system prompt"""
        return """You are an expert Research Agent. Your role is to:

1. FIND relevant information from available sources
2. EXTRACT key facts, data, and evidence
3. CITE sources accurately
4. IDENTIFY knowledge gaps
5. ORGANIZE findings logically

Research Guidelines:
- Be thorough and systematic
- Verify information from multiple angles
- Present facts objectively
- Highlight important findings
- Note any limitations or uncertainties

Output Format:
## Research Findings

### Key Facts
- [Fact 1]
- [Fact 2]
- [Fact 3]

### Detailed Analysis
[Provide detailed information and context]

### Sources
[List sources used]

### Gaps & Questions
[Note any unanswered questions or areas needing more research]
"""
    
    async def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute research task
        
        Args:
            task: Research question or topic
            context: Additional context (documents, etc.)
            
        Returns:
            Research findings with sources and analysis
        """
        context = context or {}
        
        logger.info(f"Researcher starting task: {task}")
        
        try:
            # Step 1: Search for information if vectorstore available
            search_results = []
            if self.vectorstore:
                search_results = await self._search_information(task)
            
            # Step 2: Analyze and synthesize findings
            findings = await self._analyze_findings(task, search_results, context)
            
            # Step 3: Extract key insights
            insights = await self._extract_insights(findings)
            
            self.add_to_history({
                "action": "research_completed",
                "task": task,
                "findings_length": len(findings)
            })
            
            return {
                "status": "success",
                "output": findings,
                "insights": insights,
                "sources": self._extract_sources(search_results),
                "confidence": 0.85
            }
            
        except Exception as e:
            logger.error(f"Researcher error: {e}")
            return {
                "status": "failed",
                "output": f"Research failed: {str(e)}",
                "insights": "",
                "sources": [],
                "confidence": 0.0
            }
    
    async def _search_information(self, query: str) -> List[Dict[str, Any]]:
        """Search for relevant information"""
        if not self.vectorstore:
            return []
        
        try:
            results = await self.vectorstore.search(query, top_k=5)
            
            self.add_to_history({
                "action": "search",
                "query": query,
                "results_count": len(results)
            })
            
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    async def _analyze_findings(
        self,
        task: str,
        search_results: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> str:
        """Analyze and synthesize research findings"""
        
        # Format search results
        context_text = self._format_search_results(search_results)
        
        # Additional context
        extra_context = context.get("additional_info", "")
        
        prompt = f"""{self.get_system_prompt()}

Research Task: {task}

Available Information:
{context_text}

{extra_context}

Provide comprehensive research findings:"""
        
        findings = await self.generate_response(prompt, temperature=0.5)
        
        self.add_to_history({
            "action": "analyze",
            "task": task
        })
        
        return findings
    
    async def _extract_insights(self, findings: str) -> str:
        """Extract key insights from findings"""
        
        prompt = f"""Based on these research findings, extract 3-5 key insights:

{findings}

Key Insights:"""
        
        insights = await self.generate_response(prompt, temperature=0.3)
        return insights
    
    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for prompt"""
        if not results:
            return "No search results available."
        
        formatted = []
        for i, result in enumerate(results[:5], 1):
            content = result.get("content", "")[:300]
            formatted.append(f"{i}. {content}...")
        
        return "\n\n".join(formatted)
    
    def _extract_sources(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract source references"""
        sources = []
        for result in results:
            metadata = result.get("metadata", {})
            source = metadata.get("filename", "Unknown source")
            if source not in sources:
                sources.append(source)
        return sources