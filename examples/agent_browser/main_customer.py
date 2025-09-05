# -*- coding: utf-8 -*-
"""The main entry point of the browser agent example."""

import asyncio
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env file in project root
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    print("python-dotenv not installed. Please install it with: pip install python-dotenv")
    print("Or set environment variables manually.")

from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit
from agentscope.mcp import StdIOStatefulClient
from agentscope.agent import UserAgent

from browser_agent import BrowserAgent  # pylint: disable=C0411


async def main() -> None:
    """The main entry point for the browser agent example."""
    # Setup toolkit with browser tools from MCP server
    toolkit = Toolkit()
    browser_client = StdIOStatefulClient(
        name="playwright-mcp",
        command="npx",
        args=["@playwright/mcp@latest"],
    )

    try:
        # Connect to the browser client
        await browser_client.connect()
        await toolkit.register_mcp_client(browser_client)

        # Get custom OpenAI API configuration
        api_key = os.environ.get("CUSTOM_OPENAI_API_KEY")
        base_url = os.environ.get("CUSTOM_OPENAI_BASE_URL")
        model_name = os.environ.get("CUSTOM_MODEL_NAME", "gpt-3.5-turbo")

        if api_key is None:
            raise ValueError("CUSTOM_OPENAI_API_KEY environment variable is required")
        if base_url is None:
            raise ValueError("CUSTOM_OPENAI_BASE_URL environment variable is required")
        # Create browser agent
        agent = BrowserAgent(
            name="BrowserBot",
            model=OpenAIChatModel(
                model_name=model_name,
                api_key=api_key,
                client_args={"base_url": base_url},
                stream=True,
            ),
            formatter=OpenAIChatFormatter(),
            memory=InMemoryMemory(),
            toolkit=toolkit,
            max_iters=50,
            start_url="https://www.google.com",
        )
        user = UserAgent("Bob")

        msg = None
        while True:
            msg = await user(msg)
            if msg.get_text_content() == "exit":
                break
            msg = await agent(msg)

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Cleaning up browser client...")
    finally:
        # Ensure browser client is always closed,
        # regardless of success or failure
        try:
            await browser_client.close()
            print("Browser client closed successfully.")
        except Exception as cleanup_error:
            print(f"Error while closing browser client: {cleanup_error}")


if __name__ == "__main__":
    print("Starting Browser Agent Example...")
    print(
        "The browser agent will use "
        "playwright-mcp (https://github.com/microsoft/playwright-mcp)."
        "Make sure the MCP server is can be install "
        "by `npx @playwright/mcp@latest`",
    )

    asyncio.run(main())
