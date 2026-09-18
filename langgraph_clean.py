import os
import json
import openai
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
import operator

load_dotenv()

# 1. 直接用 openai 库，指向 DeepSeek
client = openai.OpenAI(
    api_key=os.environ.get("DEEPSEEK_KEY"),
    base_url="https://api.deepseek.com"
)

# 2. 定义“行李箱”（状态）
class State(TypedDict):
    messages: Annotated[list, operator.add]

# 3. 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        }
    }
]

def get_weather(city):
    return f"{city}今天晴，25度"

tool_map = {"get_weather": get_weather}

# 4. 大模型节点
def model_node(state: State):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=state["messages"],
        tools=tools
    )
    msg = response.choices[0].message
    return {"messages": [msg]}

# 5. 工具节点
def tool_node(state: State):
    last_msg = state["messages"][-1]
    tool_call = last_msg.tool_calls[0]
    func_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    result = tool_map[func_name](**args)
    return {"messages": [{"role": "tool", "tool_call_id": tool_call.id, "content": result}]}

# 6. 路由（条件边）
def should_continue(state: State):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

# 7. 组装图
workflow = StateGraph(State)
workflow.add_node("agent", model_node)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

# 8. 加上记忆并编译
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# 9. 测试（注意 config 里的 thread_id）
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "user_001"}}
    
    res1 = app.invoke({"messages": [{"role": "user", "content": "广州天气怎么样？"}]}, config)
    print("第一轮回答：", res1["messages"][-1].content)
    
    res2 = app.invoke({"messages": [{"role": "user", "content": "那刚才那个城市适合穿外套吗？"}]}, config)
    print("第二轮回答：", res2["messages"][-1].content)