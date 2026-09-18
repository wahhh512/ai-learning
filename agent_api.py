from fastapi import FastAPI
from pydantic import BaseModel
import os, json, openai
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

client = openai.OpenAI(
    api_key=os.environ.get("DEEPSEEK_KEY"),
    base_url="https://api.deepseek.com"
)

# 工具定义和真实函数（复制你之前的 tools、tool_map、get_weather、calculate）
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


tool_map = {
    "get_weather": get_weather,
        "calculate": calculate
}
chat_history = []

class ChatRequest(BaseModel):
    message: str

@app.post("/agent_chat")
def agent_chat(req: ChatRequest):
    chat_history.append({"role": "user", "content": req.message})
    response = client.chat.completions.create(
        model="deepseek-chat", messages=chat_history, tools=tools
    )
    msg = response.choices[0].message
    chat_history.append(msg)
    
    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        result = tool_map[func_name](**args)
        
        chat_history.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        final = client.chat.completions.create(
            model="deepseek-chat", messages=chat_history
        )
        chat_history.append(final.choices[0].message)
        return {"reply": final.choices[0].message.content, "used_tool": func_name}
    
    return {"reply": msg.content, "used_tool": None}