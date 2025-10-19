from typing import Dict, Any
from app.services.agents.react_agent import BaseAgent
from app.models.rag_model import ReActStep, AgentType
from app.services.prompt_template import WriterPromptTemplate

class WriterAgent(BaseAgent):
    def __init__(self, groq_service):
        super().__init__(AgentType.WRITER, groq_service)
        self.prompt_template = WriterPromptTemplate()
    
    async def think(self, query: str, context: Dict[str, Any]) -> ReActStep:
        research_data = context.get("research_data", "")
        
        prompt = self.prompt_template.create_think_prompt(query, research_data)
        response = await self.groq_service.generate_response(prompt)
        thought = response.choices[0].message.content
        
        return ReActStep(
            thought=thought,
            action="synthesize",
            action_input={"query": query, "research_data": research_data}
        )
    
    async def act(self, action: str, action_input: Dict[str, Any]) -> str:
        if action == "synthesize":
            prompt = self.prompt_template.create_synthesize_prompt(
                action_input["query"],
                action_input["research_data"]
            )
            response = await self.groq_service.generate_response(prompt)
            return response.choices[0].message.content
        else:
            return f"Unknown action: {action}"