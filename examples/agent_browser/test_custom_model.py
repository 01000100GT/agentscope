#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试自定义 OpenAI 兼容模型的简单脚本"""

import asyncio
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env file in project root
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)
    print(f"已加载环境变量文件: {env_path}")
except ImportError:
    print("python-dotenv not installed. Please install it with: pip install python-dotenv")
    print("Or set environment variables manually.")

from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.agent import ReActAgent
from agentscope.message import Msg


async def test_custom_model():
    """测试自定义 OpenAI 兼容模型"""
    
    print("=" * 60)
    print("测试自定义 OpenAI 兼容模型")
    print("=" * 60)
    
    # Get custom OpenAI API configuration
    api_key = os.environ.get("CUSTOM_OPENAI_API_KEY")
    base_url = os.environ.get("CUSTOM_OPENAI_BASE_URL")
    model_name = os.environ.get("CUSTOM_MODEL_NAME", "gpt-3.5-turbo")
    
    print(f"API Key: {api_key[:10]}..." if api_key else "未设置")
    print(f"Base URL: {base_url}")
    print(f"Model Name: {model_name}")
    print("-" * 60)
    
    if api_key is None:
        raise ValueError("CUSTOM_OPENAI_API_KEY environment variable is required")
    if base_url is None:
        raise ValueError("CUSTOM_OPENAI_BASE_URL environment variable is required")
    
    try:
        # Create custom model
        model = OpenAIChatModel(
            model_name=model_name,
            api_key=api_key,
            client_args={"base_url": base_url},
            stream=True,
        )
        
        # Create agent
        agent = ReActAgent(
            name="CustomModelBot",
            sys_prompt="你是一个智能助手，请简洁地回答用户的问题。",
            model=model,
            formatter=OpenAIChatFormatter(),
            memory=InMemoryMemory(),
        )
        
        # Test message
        test_msg = Msg("用户", "你好！请简单介绍一下你自己。", "user")
        
        print("发送测试消息...")
        response = await agent(test_msg)
        
        print("\n✅ 模型响应成功!")
        print(f"回答: {response.get_text_content()}")
        print("\n" + "=" * 60)
        print("✅ 自定义模型配置正确，可以正常使用！")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 模型测试失败: {str(e)}")
        print("\n请检查以下配置:")
        print("1. .env 文件中的 CUSTOM_OPENAI_API_KEY 是否正确")
        print("2. .env 文件中的 CUSTOM_OPENAI_BASE_URL 是否正确")
        print("3. .env 文件中的 CUSTOM_MODEL_NAME 是否正确")
        print("4. 网络连接是否正常")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_custom_model())
    if success:
        print("\n🎉 现在可以继续设置 Node.js 来使用浏览器功能了！")
    else:
        print("\n⚠️  请先解决模型配置问题，再进行下一步。")