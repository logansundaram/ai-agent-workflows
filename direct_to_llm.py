from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from secret_config import character


llm = ChatOllama(model="Qwen2.5-coder:14B", temperature=0)


sys_msg = SystemMessage(content=character)


message = [
    (
        sys_msg
    ),
    ("human", "hi"),
]

llm_response = llm.invoke(message)

print(llm_response)