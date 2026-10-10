"""An MCP server exposing Muse's content tools to any MCP-capable client or agent."""
import sys
sys.path.insert(0, "/home/claude/gb/ch15")
from mcp.server.mcpserver import MCPServer
import tools as T
server = MCPServer("northwind-content")

@server.tool()
def search_guidelines(question: str) -> dict:
    """Answer a question from Northwind brand guidelines, with a citation, or say it is not covered."""
    return T.run_tool("search_guidelines", {"question": question})

@server.tool()
def check_copy(product: str, headline: str, body: str, hashtags: list[str], features: list[str]) -> dict:
    """Check a social post against Northwind brand rules; returns a list of problems (empty if fine)."""
    return T.run_tool("check_copy", dict(product=product, headline=headline, body=body, hashtags=hashtags, features=features))

@server.tool()
def list_schedule(start: str, end: str) -> dict:
    """List scheduled posts between two ISO dates (YYYY-MM-DD)."""
    return T.run_tool("list_schedule", {"start": start, "end": end})

if __name__ == "__main__":
    server.run()   # stdio transport. schedule_post is deliberately NOT exposed: publishing stays behind Muse's approval flow.
