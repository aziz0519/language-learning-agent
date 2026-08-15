import asyncio
import os
from pathlib import Path

from typing import TypedDict, Annotated, Optional

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient

from dotenv import load_dotenv

from agent.tools import get_n_random_words, get_n_random_words_by_difficulty_level, translate_words



load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    source_language: Optional[str]
    number_of_words: Optional[int]
    word_difficulty: Optional[str]
    target_language: Optional[str]


def clanki_js_path() -> str:
    """
    Resolve the path to the Clanki MCP server entry point.
    
    Tries the following in order:
    1. Environment variable CLANKI_JS_PATH
    2. Repository-relative path: ../clanki/build/index.js
    
    Raises FileNotFoundError if the path cannot be resolved and validated.
    """
    # Try environment variable first
    env_path = os.getenv("CLANKI_JS_PATH")
    if env_path:
        path_obj = Path(env_path).expanduser()
        if path_obj.exists():
            return str(path_obj.resolve())
        raise FileNotFoundError(
            f"CLANKI_JS_PATH environment variable points to non-existent path: {env_path}"
        )
    
    # Try repository-relative path
    current_dir = Path(__file__).resolve().parent
    repo_relative = current_dir.parent / "clanki" / "build" / "index.js"
    if repo_relative.exists():
        return str(repo_relative)
    
    raise FileNotFoundError(
        f"Could not find Clanki MCP server. "
        f"Set CLANKI_JS_PATH environment variable or place Clanki at {repo_relative}"
    )


CLANKI_JS = clanki_js_path()

local_tools = [
    get_n_random_words,
    get_n_random_words_by_difficulty_level,
    translate_words
]


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
    mcp_tools = await client.get_tools()
    return [*local_tools, *mcp_tools]

def assistant(state: AgentState):
    textual_description_of_tools = """
    def get_n_random_words(language:str, n:int) -> list:
    Get a specified number of random words from a word list for a given language.
    """
    sys_msg = SystemMessage(content=f"""
    You are a helpful language learning assistant. You can carry out actions using the following tools:{textual_description_of_tools}
    
    The user is going to give you a command.
    
    Your job is to check:
    1. Which source language that the user wants words from.
    2. How many word they want.
    3. Whether they want words of a specific difficulty, part-of-speech, or just random words.
    4. Whether they want these words translated into a target language. 
    5. Whether they want to add these words to an Anki deck. Make sure the `create-deck` tool is called before `create-card`
    """)

    tools = assistant.tools if hasattr(assistant, "tools") else []
    llm = ChatOpenAI(model="gpt-4o")
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

    return {
        "messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]

    }

async def build_graph():
    """Build the state graph with properly initialized tools."""
    tools = await setup_tools()
    assistant.tools = tools

    builder = StateGraph(AgentState)

    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition
    )
    
    builder.add_edge("tools", "assistant")
    
    return builder.compile()


async def main():
    """Main async function to run the application."""
    react_graph = await build_graph()
    
    user_prompt = "Please get 10 easy words in Spanish, translate them to English, and create a new Anki deck with them called Spanish::Easy."
    
    messages = [HumanMessage(content=user_prompt)]
    
    result = await react_graph.ainvoke({
        "messages": messages,
        "source_language": None,
        "number_of_words": None,
        "target_language": None,
        "word_difficulty": None
    })
    
    print(f"Final messages: {result['messages'][-1].content}")
    
if __name__ == "__main__":
    asyncio.run(main())