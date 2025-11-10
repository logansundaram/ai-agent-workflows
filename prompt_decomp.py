from langchain_ollama import ChatOllama
import json

llm = ChatOllama(model="gpt-oss:20B", temperature=0)

# ---------------- Prompt Decomposition ----------------
def decompose_prompt(user_request: str) -> list[dict]:
    """
    Ask the LLM to decompose the user's request into smaller tasks.
    Returns a list of {id, description} dicts.
    """

    decomp_messages = [
        (
            "system",
            "You are a planning assistant. Your job is to break a user request "
            "into a small, ordered list of concrete coding tasks.\n"
            "Return ONLY valid JSON, no extra text. Format:\n"
            "[{\"id\": 1, \"description\": \"...\"}, {\"id\": 2, \"description\": \"...\"}]"
        ),
        (
            "human",
            f"Decompose this request into 2–6 clear coding subtasks:\n\n{user_request}"
        ),
    ]

    decomp_response = llm.invoke(decomp_messages)
    # LangChain ChatOllama returns an AIMessage; the text is in .content
    raw = decomp_response.content if hasattr(decomp_response, "content") else str(decomp_response)

    try:
        subtasks = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: wrap in a single task if JSON got messed up
        subtasks = [{"id": 1, "description": user_request}]

    # Ensure it's a list of dicts
    if not isinstance(subtasks, list):
        subtasks = [{"id": 1, "description": user_request}]

    return subtasks

# ---------------- Solver for each subtask ----------------
def solve_subtask(subtask: dict) -> str:
    """
    Use the same LLM with your 'coding assistant' system prompt
    to solve each subtask.
    """
    task_desc = subtask["description"]

    messages = [
        (
            "system",
            "You are a helpful assistant that excels at understanding and writing code. "
            "Explain your reasoning concisely and give correct, runnable Python."
        ),
        (
            "human",
            f"Subtask {subtask['id']}:\n{task_desc}"
        ),
    ]

    resp = llm.invoke(messages)
    return resp.content if hasattr(resp, "content") else str(resp)

# ---------------- Full pipeline ----------------
if __name__ == "__main__":
    user_request = "write me a python function that determines if a number is prime or not"

    # 1) Decompose the prompt
    subtasks = decompose_prompt(user_request)
    print("Decomposed subtasks:\n")
    for t in subtasks:
        print(f"- [{t['id']}] {t['description']}")
    print("\n" + "=" * 60 + "\n")

    # 2) Solve each subtask
    answers = []
    for t in subtasks:
        print(f"--- Solving subtask {t['id']} ---")
        ans = solve_subtask(t)
        answers.append((t, ans))
        print(ans)
        print("\n" + "-" * 40 + "\n")

    # 3) Optionally merge answers into a final response
    merge_messages = [
        (
            "system",
            "You are a senior engineer. You will receive a list of subtasks and their draft answers. "
            "Combine them into a single, clean final answer for the original request."
        ),
        (
            "human",
            json.dumps(
                {
                    "original_request": user_request,
                    "subtasks_and_answers": [
                        {"id": t["id"], "description": t["description"], "answer": ans}
                        for t, ans in answers
                    ],
                },
                indent=2,
            ),
        ),
    ]

    final_response = llm.invoke(merge_messages)
    final_text = final_response.content if hasattr(final_response, "content") else str(final_response)

    print("\nFINAL COMBINED ANSWER:\n")
    print(final_text)