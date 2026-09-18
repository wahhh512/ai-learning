import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # 1. 告诉客户端怎么启动 MCP Server（和你 mcp.json 里配置的一样）
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"]
    )
    
    # 2. 建立连接（stdio 即标准输入输出，本地子进程通信）
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 3. 初始化握手
            await session.initialize()
            
            # 4. 获取工具列表
            tools = await session.list_tools()
            print("MCP Server 暴露的工具：", [t.name for t in tools.tools])
            
            # 5. 调用 add 工具（用代码传参）
            result = await session.call_tool("add", arguments={"a": 10, "b": 20})
            print("add(10, 20) 返回结果：", result.content[0].text)
            
            # 6. 调用 get_weather 工具
            weather = await session.call_tool("get_weather", arguments={"city": "广州"})
            print("get_weather(广州) 返回结果：", weather.content[0].text)

if __name__ == "__main__":
    asyncio.run(main())