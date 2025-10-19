from typing import Dict, Any
from app.services.agents.react_agent import BaseAgent
from app.models.rag_model import ReActStep, AgentType
from app.services.vectorstore import VectorStore
from app.services.prompt_template import ResearcherPromptTemplate

class ResearcherAgent(BaseAgent):
    def __init__(self, groq_service, vector_store: VectorStore):
        super().__init__(AgentType.RESEARCHER, groq_service)
        self.vector_store = vector_store
        self.prompt_template = ResearcherPromptTemplate()
    
    async def think(self, query: str, context: Dict[str, Any]) -> ReActStep:
        prompt = self.prompt_template.create_think_prompt(query, context)
        
        response = await self.groq_service.generate_response(prompt)
        
        thought = response.choices[0].message.content
        
        return ReActStep(
            thought=thought,
            action="search",
            action_input={"query": query, "filters": context.get("filters", {})}
        )
    
    async def act(self, action: str, action_input: Dict[str, Any]) -> str:
        if action == "search":
            results = await self.vector_store.similarity_search(
                action_input["query"],
                k=action_input.get("k", 3)
            )
            return "\n".join([f"Source {i+1}: {doc.page_content}" 
                            for i, doc in enumerate(results)])
        else:
            return f"Unknown action: {action}"