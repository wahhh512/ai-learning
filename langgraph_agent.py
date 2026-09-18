import os, json, openai
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
import operator

load_dotenv()

client = openai.OpenAI(
    api_key=os.environ.get("DEEPSEEK_KEY"),
    base_url="https://api.deepseek.com"
)

# 1. 定义状态（这是图的“行李箱”）
class State(TypedDict):
    messages: Annotated[list, operator.add]  # operator.add 表示每次追加，而不是覆盖

# 2. 定义工具（复制昨天的）
tools = [
    {"type": "function", "function": {"name": "get_weather", "description": "获取指定城市的天气", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "calculate", "description": "做数学计算", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}
]

def get_weather(city):
    return f"{city}今天晴，25度"

def calculate(expression):
    return str(eval(expression))

tool_map = {"get_weather": get_weather, "calculate": calculate}

# 3. 定义调用大模型的节点
def model_node(state: State):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=state["messages"],
        tools=tools
    )
    msg = response.choices[0].message
    return {"messages": [msg]}

# 4. 定义执行工具的节点
def tool_node(state: State):
    messages = state["messages"]
    last_msg = messages[-1]
    tool_call = last_msg.tool_calls[0]
    func_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    
    result = tool_map[func_name](**args)
    
    # 返回工具执行结果，把 tool_call_id 带上，这是规范
    return {"messages": [{"role": "tool", "tool_call_id": tool_call.id, "content": result}]}

# 5. 定义路由逻辑：判断模型是决定调工具，还是已经给出了最终回答
def should_continue(state: State):
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"  # 如果有工具调用，走工具节点
    return END          # 如果没有，流程结束

# 6. 构建图
workflow = StateGraph(State)
workflow.add_node("agent", model_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent") # 工具执行完，回到模型节点继续总结

# 7. 加上记忆
memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)

# 8. 测试
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "user_001"}}  # 用户唯一标识，解决串台和失忆
    
    # 第一轮
    res1 = app_graph.invoke({"messages": [{"role": "user", "content": "广州天气怎么样？"}]}, config)
    print("第一轮回答：", res1["messages"][-1].content)
    
    # 第二轮（考验记忆）
    res2 = app_graph.invoke({"messages": [{"role": "user", "content": "那刚才那个城市适合穿外套吗？"}]}, config)
    print("第二轮回答：", res2["messages"][-1].content)