def build_prompt(context: str, query: str, chat_history: list[dict]) -> str:
    # Build a readable chat history text
    history_text = ""
    for turn in chat_history:
        history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

    prompt = f"""
You are an intelligent assistant. 
Answer questions using the provided context and chat history. 
If the context does not contain the answer, respond clearly that you don't know.

--- Chat History ---
{history_text if history_text else "No previous conversation."}

--- Context ---
{context}

--- Question ---
{query}

--- Instructions ---
- Be clear and helpful
- Use a professional tone
- Do not hallucinate
- If unsure, say: "I don't have enough information to answer."

Answer:
    """
    return prompt.strip()
