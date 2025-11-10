from langchain_ollama import ChatOllama

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode


llm = ChatOllama(model="gpt-oss:20b", temperature=0)

def multiply(a: int, b: int) -> int:
    return a * b

def add(a: int, b: int) -> int:
    return a + b

tools = [add, multiply]
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="You are a helpful assistant who can perfom arithmetic on integers")

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


builder = StateGraph(MessagesState)


builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition,)

builder.add_edge("tools", "assistant")

react_graph = builder.compile()

