# from groq import Groq
# from app.core.config import settings

# client = Groq(api_key=settings.GROQ_API_KEY)

# def generate_answer(query: str, contexts: list[str]) -> str:
#     context_text = "\n\n".join(contexts) if contexts else "No context found."
#     prompt = f"Answer the question based on the context below:\n\nContext:\n{context_text}\n\nQuestion: {query}\nAnswer:"

#     response = client.chat.completions.create(
#         model=settings.GROQ_MODEL,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.2,
#     )

#     # ✅ Access content as attribute, not dictionary
#     return response.choices[0].message.content




from groq import Groq
from app.core.config import settings
from app.services.prompt_template import build_prompt

client = Groq(api_key=settings.GROQ_API_KEY)

def generate_answer_with_history(query: str, contexts: list[str], chat_history: list[dict]) -> str:
    context_text = "\n\n".join(contexts) if contexts else "No context found."
    prompt = build_prompt(context_text, query, chat_history)

    response = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content
