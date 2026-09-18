import os
import json
import openai
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(
    api_key=os.environ.get("DEEPSEEK_KEY"),
    base_url="https://api.deepseek.com"
)

# 定义两个工具（模型能调用的函数）
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "做数学计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"]
            }
        }
    }
]

# 模拟工具的实际执行函数
def get_weather(city):
    return f"{city}今天晴，25度"

def calculate(expression):
    return str(eval(expression))

# 工具名到函数的映射
tool_map = {
    "get_weather": get_weather,
    "calculate": calculate
}


# 在函数外面定义全局历史
chat_history = []

def run_agent(user_input):
    messages = [{"role": "user", "content": user_input}]
    
    # 第一次调用：让模型决定是否要调工具
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools
    )
    
    msg = response.choices[0].message
    chat_history.append(msg)
    
    # 如果模型决定调用工具
    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        
        print(f"模型决定调用工具：{func_name}，参数：{args}")
        
        # 执行工具
        result = tool_map[func_name](**args)
        print(f"工具返回结果：{result}")
        
        # 把工具结果发回给模型，让它生成最终回答
        messages.append(msg)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })

        chat_history.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })
        
        final = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )
        return final.choices[0].message.content
    else:
        return msg.content

if __name__ == "__main__":
    print(run_agent("广州天气怎么样？"))
    print("---")
    print(run_agent("那刚才那个城市适合穿外套吗？"))