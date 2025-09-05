#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""不需要 Node.js 的浏览器智能体示例"""

import asyncio
import os
from pathlib import Path

# 直接设置环境变量
os.environ["CUSTOM_OPENAI_API_KEY"] = "sk-sbwxkcjoeolowiokiusntqcqjbwqtyqlzrlxvjfykpspliqd"
os.environ["CUSTOM_OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"
os.environ["CUSTOM_MODEL_NAME"] = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.agent import ReActAgent, UserAgent
from agentscope.message import Msg


async def main():
    """不依赖浏览器工具的智能体示例"""
    
    print("🚀 启动自定义模型智能体（不需要浏览器工具）...")
    
    # Get custom OpenAI API configuration
    api_key = os.environ.get("CUSTOM_OPENAI_API_KEY")
    base_url = os.environ.get("CUSTOM_OPENAI_BASE_URL")
    model_name = os.environ.get("CUSTOM_MODEL_NAME", "gpt-3.5-turbo")
    
    if not api_key or not base_url:
        print("❌ 环境变量未设置正确")
        return
    
    try:
        # Create custom model
        model = OpenAIChatModel(
            model_name=model_name,
            api_key=api_key,
            client_args={"base_url": base_url},
            stream=True,
        )
        
        # Create agent without browser tools
        agent = ReActAgent(
            name="智能助手",
            sys_prompt="你是一个智能助手，可以回答各种问题和进行对话。",
            model=model,
            formatter=OpenAIChatFormatter(),
            memory=InMemoryMemory(),
        )
        
        user = UserAgent("用户")
        
        print("✅ 智能体已启动！")
        print("💡 输入 'exit' 退出程序")
        print("=" * 50)
        
        msg = None
        while True:
            # 获取用户输入
            user_input = input("👤 您: ")
            if user_input.lower() == "exit":
                print("👋 再见！")
                break
                
            # 创建消息
            msg = Msg("用户", user_input, "user")
            
            # 获取智能体响应
            print("🤖 助手: ", end="", flush=True)
            response = await agent(msg)
            print(response.get_text_content())
            print("-" * 30)
            
    except Exception as e:
        print(f"❌ 运行出错: {e}")


if __name__ == "__main__":
    print("自定义模型智能体示例 - 无需浏览器工具")
    print("使用模型: DeepSeek-R1-Distill-Qwen-7B")
    print("=" * 50)
    
    asyncio.run(main())