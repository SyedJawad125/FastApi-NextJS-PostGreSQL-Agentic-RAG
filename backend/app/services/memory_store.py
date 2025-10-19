from typing import Dict, List

# Simple in-memory memory store
# Example: memory["session_id"] = [{"user": "...", "assistant": "..."}]
memory: Dict[str, List[Dict[str, str]]] = {}

def get_history(session_id: str) -> List[Dict[str, str]]:
    return memory.get(session_id, [])

def add_to_history(session_id: str, user_msg: str, assistant_msg: str):
    if session_id not in memory:
        memory[session_id] = []
    memory[session_id].append({"user": user_msg, "assistant": assistant_msg})
