import os
import shutil
from dotenv import load_dotenv
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage 
from haystack_integrations.tools.mcp import MCPTool, StdioServerInfo
from haystack.components.agents import Agent

 
class SafeMCPTool(MCPTool):
    def __deepcopy__(self, memo):
        return memo # Do no copy, reuse the instance

load_dotenv()

github_token = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
if not github_token:
    raise ValueError("GITHUB_PERSONAL_ACCESS_TOKEN environment variable not set.")

github_repo_name = os.getenv('GITHUB_REPO_NAME')
if not github_repo_name:
    raise ValueError("GITHUB_REPO_NAME environment variable not set.")

has_docker = shutil.which("which docker")
github_mcp_server_env={
    "GITHUB_PERSONAL_ACCESS_TOKEN": github_token, 
    "GITHUB_DYNAMIC_TOOLSETS": "1"
}
if has_docker:
    github_mcp_server = StdioServerInfo(
        command="docker",
        args=[
            "run",
            "--rm",
            "-p", "8080:8080",
            "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
            "-e", "GITHUB_TOOLSETS",
            "modelcontextprotocol/server-github"
        ],
        env=github_mcp_server_env,
    )
else: 
    github_mcp_server = StdioServerInfo(
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-github"
        ],
        env=github_mcp_server_env,
    )

print("MCP server is created")

get_github_file_contents_tool = SafeMCPTool(name="get_file_contents", server_info=github_mcp_server, description="Get contents of a file or directory from Github")
create_github_issue_tool = SafeMCPTool(name="create_issue", server_info=github_mcp_server, description="Create a new issue in a GitHub repository")

tools = [get_github_file_contents_tool, create_github_issue_tool]

print("MCP tools are created")

# There is a bug in the MCPTool use of asyncio that prevents it from being used as a tool in the agent.
# Thats why I had to create SafeMCPTool to prevent errores due to deepcopying.
agent = Agent(
    chat_generator=OpenAIChatGenerator(),
    system_prompt="""
    You are a helpful Agent that can read Github repositories content 
    and create issues.
    """,
    tools=tools,
)

print("Agent created")

## Agent query
user_input = f"""
Look for typos in the README.md of the Github repository {github_repo_name}.
If any real ortographic typo is found then create one issue (labels "typo", "docs") to fix them: 
specify the line and column. Return the number of typos found (0 if none) in raw 
JSON without backticks nor formatting.
<exemple>
{{
    "typos_count": 1
}}
</exemple>
"""

response = agent.run(messages=[ChatMessage.from_user(user_input)])

## Print the agent thinking process
# print(response)

## Print the final response
print(response["messages"][-1].text)
