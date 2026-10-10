import asyncio, json
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
async def main():
    async with stdio_client(StdioServerParameters(command="python3", args=["muse_mcp_server.py"])) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize(); tools = (await s.list_tools()).tools
            for t in tools: print("TOOL", t.name, "| required:", t.input_schema.get("required"))
            out = await s.call_tool("check_copy", {"product": "TrailMug", "headline": "Best ever mug!", "body": "TrailMug keeps drinks hot for 12 hours.",
                                                  "hashtags": ["#Northwind"], "features": ["keeps drinks hot for 6 hours"]})
            print("RESULT", out.content[0].text)
            out = await s.call_tool("list_schedule", {"start": "2026-10-13", "end": "2026-10-14"}); print("RESULT", out.content[0].text)
asyncio.run(main())
