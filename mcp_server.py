import sys
import logging

# 将日志输出到 stderr，避免污染 stdout 的 JSON-RPC 通信
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

from mcp.server.fastmcp import FastMCP

# 创建 MCP 服务端，给它起个名字
mcp = FastMCP("Demo")

# 用装饰器标记这是一个工具，AI 可以调用
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    return f"{city}今天晴，25度"

# 用装饰器标记这是一个资源，AI 可以读取
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run()