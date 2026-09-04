from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv(override=True)

import asyncio

async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["mathserver.py"],
                "transport": "stdio"
            },
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http"
            }
        }
    )

    MODEL = os.getenv("OLLAMA_MODEL")

    tools = await client.get_tools()
    model = init_chat_model(
        model=MODEL
    )

    agent = create_react_agent(
        model,
        tools
    )

    math_response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What's (3 + 5) * 10?"
                }
            ]
        }
    )

    print("Math Response: ", math_response["messages"][-1].content)

    weather_response = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the weather in Delhi?"
                    }
                ]
            }
        )
    
    print("Weather Response: ", weather_response["messages"][-1].content)
    


asyncio.run(main())