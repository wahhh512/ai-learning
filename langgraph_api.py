from fastapi import FastAPI
from pydantic import BaseModel
import os, json, openai
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
import operator

load_dotenv()
app = FastAPI()

client = openai.OpenAI(
    api_key=os.environ.get("DEEPSEEK_KEY"),
    base_url="https://api.deepseek.com"
)

class State(TypedDict):
    messages: Annotated[list, operator.add]

tools = [
    {"type": "function", "function": {"name": "get_weather", "description": "获取指定城市的天气", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
]

def get_weather(city):
    return f"{city}今天晴，25度"

tool_map = {"get_weather": get_weather}

def model_node(state: State):
    response = client.chat.completions.create(
        model="deepseek-chat", messages=state["messages"], tools=tools
    )
    return {"messages": [response.choices[0].message]}

def tool_node(state: State):
    last_msg = state["messages"][-1]
    tool_call = last_msg.tool_calls[0]
    func_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    result = tool_map[func_name](**args)
    return {"messages": [{"role": "tool", "tool_call_id": tool_call.id, "content": result}]}

def should_continue(state: State):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

workflow = StateGraph(State)
workflow.add_node("agent", model_node)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")
memory = MemorySaver()
agent_app = workflow.compile(checkpointer=memory)

class ChatRequest(BaseModel):
    message: str
    thread_id: str  # 让前端传用户ID，用于隔离记忆

@app.post("/agent_chat")
def agent_chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = agent_app.invoke(
        {"messages": [{"role": "user", "content": req.message}]},
        config
    )
    return {"reply": result["messages"][-1].content}