from typing import TypedDict, Annotated

from langchain_core.messages import AnyMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient

from dotenv import load_dotenv

from agent.tools import {
    get_n_random_words,
}

load_dotenv()

class AgentState(TypedDict):
    message: Annotated[list[AnyMessage], add_messages]


local_tools = [
    get_n_random_words,
]

CLANKI_JS = "/Users/Jodie.Burchell/Documents/git/clanki/build/index.js"


async def setup_tools():
    client = MultiServerMCPClient(
        {
            "clanki": {
                "command":"node",
                "args": [CLANKI_JS],
                "transport": "stdio",
            }
        }
    )

def assistant(state: AgentState):
    sys_msg = SystemMessage(content=f"""
    You are a helpful language learning assistant. 
    
    The user is going to give you a command.
    """)

    tools = assistant.tools if hasattr(assistant, "tools") else []
    llm = ChatOpenAI(model="gpt-4o")
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

    return {
        "messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]

    }