from langchain_ollama import ChatOllama

llm = ChatOllama(model="Qwen2.5-coder:14B", temperature=0)

message = [
    (
        "system",
        "You are a helpful assistant that excels at understanding and writing code.",
    ),
    ("human", "write me a python function that determines if a number is prime or not"),
]

llm_response = llm.invoke(message)

print(llm_response)