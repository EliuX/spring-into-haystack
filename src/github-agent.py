import os
import shutil
from typing import List
from dotenv import load_dotenv
from haystack import Pipeline, component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage 
from haystack_integrations.tools.mcp import MCPTool, StdioServerInfo
from haystack.components.tools import ToolInvoker   
from haystack.components.converters import OutputAdapter
        
@component
class MessageRouter:
    @component.output_types(nextAction=List[ChatMessage], forAnalysis=List[ChatMessage], response=List[ChatMessage])
    def run(self, 
            from_user: List[ChatMessage] = None,
            tool_messages: List[ChatMessage] = None,
            messages: List[ChatMessage] = None):
        msg = messages or from_user or tool_messages 
        if msg[0].tool_calls:
            print("- Tool call detected: ", msg[0])
            return {"nextAction": msg}
        elif from_user or tool_messages or (messages and msg[0].from_tool):
            print("- Content to be analyzed: ", msg[0])
            return {"forAnalysis": msg}          
        else:
            print("- Final content: ", msg[0])
            return {"response": msg}   

load_dotenv()

if not shutil.which("docker"):
    raise RuntimeError("Required docker command not found")

github_token = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
if not github_token:
    raise ValueError("GITHUB_PERSONAL_ACCESS_TOKEN environment variable not set.")

github_repo_name = os.getenv('GITHUB_REPO_NAME')
if not github_repo_name:
    raise ValueError("GITHUB_REPO_NAME environment variable not set.")

github_mcp_server = StdioServerInfo(
    command="npx",
    args=[
        "-y",
        "@modelcontextprotocol/server-github"
    ],
    env={
        "GITHUB_PERSONAL_ACCESS_TOKEN": github_token, 
        "GITHUB_TOOLSETS": "all"
    },
)

print("MCP server is created")

get_github_file_contents_tool = MCPTool(name="get_file_contents", server_info=github_mcp_server, description="Get contents of a file or directory from Github")
create_github_issue_tool = MCPTool(name="create_issue", server_info=github_mcp_server, description="Create a new issue in a GitHub repository")

tools = [get_github_file_contents_tool, create_github_issue_tool]

print("MCP tools are created")

# There is a bug in the MCPTool use of asyncio that prevents it from being used as a tool in the agent.
# agent = Agent(
#     chat_generator=OpenAIChatGenerator(),
#     system_prompt="""
#     You are a helpful Agent that can read Github repositories content 
#     and create issues.
#     """,
#     tools=tools,
#     exit_conditions=["create_issue"],
# )

agent = Pipeline()
agent.add_component("llm", OpenAIChatGenerator(tools=tools))
agent.add_component("tool_invoker", ToolInvoker(tools=tools)) 
agent.add_component("router", MessageRouter())
# agent.add_component(
#     "adapter",
#     OutputAdapter(
#         template="{{ tool_messages }}",
#         output_type=list[ChatMessage],
#         unsafe=True,
#     ),
# )


agent.connect("router.forAnalysis", "llm.messages")
agent.connect("llm.replies", "router.messages")
agent.connect("router.nextAction", "tool_invoker.messages")
agent.connect("tool_invoker.tool_messages", "router.tool_messages")

agent.draw("logo/agent.png") # Check on the design


print("Agent created")

## Query to test agent
user_input = f"""
Fetch for the content of the README.md of the Github repository {github_repo_name}, if not available.
If the content is available look for typos. If typos are found then create an issue in the repository.
After the issue is created, return the number of typos found (0 if none).
"""

response = agent.run({
    "router": {
        "from_user": [ChatMessage.from_user(user_input)]
    }
})

## Print the agent thinking process
print(response)

# Print the final response
# for msg in response["response"]:
#     print(msg.text)
