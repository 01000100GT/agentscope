#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试自定义模型与Deep Research Agent的兼容性"""

import asyncio
import os
from pathlib import Path

# 直接设置环境变量，不依赖 dotenv
os.environ["CUSTOM_OPENAI_API_KEY"] = "sk-sbwxkcjoeolowiokiusntqcqjbwqtyqlzrlxvjfykpspliqd"
os.environ["CUSTOM_OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"
os.environ["CUSTOM_MODEL_NAME"] = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.message import Msg


async def test_simple_conversation():
    """测试简单对话以验证模型基本功能"""
    print("🔍 测试简单对话...")
    
    # Get custom OpenAI API configuration
    api_key = os.environ.get("CUSTOM_OPENAI_API_KEY")
    base_url = os.environ.get("CUSTOM_OPENAI_BASE_URL")
    model_name = os.environ.get("CUSTOM_MODEL_NAME", "gpt-3.5-turbo")

    if api_key is None:
        raise ValueError("CUSTOM_OPENAI_API_KEY environment variable is required")
    if base_url is None:
        raise ValueError("CUSTOM_OPENAI_BASE_URL environment variable is required")

    model = OpenAIChatModel(
        model_name=model_name,
        api_key=api_key,
        client_args={"base_url": base_url},
        stream=False,  # 使用非流式以简化测试
    )
    
    formatter = OpenAIChatFormatter()
    
    # 测试简单的系统提示和用户消息
    messages = [
        Msg("system", "You are a helpful assistant.", "system"),
        Msg("user", "请简单回答：1+1等于几？", "user")
    ]
    
    try:
        formatted_msgs = await formatter.format(messages)
        print("✅ 消息格式化成功")
        print(f"格式化后的消息: {formatted_msgs}")
        
        response = await model(formatted_msgs)
        print("✅ 模型调用成功")
        
        if hasattr(response, 'content') and response.content:
            for block in response.content:
                block_dict = dict(block)
                if block_dict.get("type") == "text":
                    print(f"✅ 模型响应: {block_dict.get('text', '')}")
                    break
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


async def test_conversation_with_multiple_messages():
    """测试多轮对话消息序列"""
    print("\n🔍 测试多轮对话消息序列...")
    
    api_key = os.environ.get("CUSTOM_OPENAI_API_KEY")
    base_url = os.environ.get("CUSTOM_OPENAI_BASE_URL")
    model_name = os.environ.get("CUSTOM_MODEL_NAME", "gpt-3.5-turbo")

    model = OpenAIChatModel(
        model_name=model_name,
        api_key=api_key,
        client_args={"base_url": base_url},
        stream=False,
    )
    
    formatter = OpenAIChatFormatter()
    
    # 模拟 Deep Research Agent 可能的消息序列
    messages = [
        Msg("system", "You are a helpful research assistant.", "system"),
        Msg("user", "请研究一下人工智能的发展历史", "user"),
        Msg("assistant", "我将帮助您研究人工智能的发展历史。", "assistant"),
        Msg("user", "请从1950年代开始说起", "user")
    ]
    
    try:
        formatted_msgs = await formatter.format(messages)
        print("✅ 多轮消息格式化成功")
        
        response = await model(formatted_msgs)
        print("✅ 多轮对话模型调用成功")
        
        if hasattr(response, 'content') and response.content:
            for block in response.content:
                block_dict = dict(block)
                if block_dict.get("type") == "text":
                    text = block_dict.get('text', '')
                    print(f"✅ 模型响应: {text[:100]}...")
                    break
        
        return True
        
    except Exception as e:
        print(f"❌ 多轮对话测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("🚀 开始测试自定义模型兼容性...")
    print("=" * 60)
    
    test1_success = await test_simple_conversation()
    test2_success = await test_conversation_with_multiple_messages()
    
    print("\n" + "=" * 60)
    if test1_success and test2_success:
        print("✅ 所有测试通过！自定义模型兼容性良好。")
        print("💡 如果 Deep Research Agent 仍然有问题，可能需要调整系统提示符或消息格式。")
    else:
        print("❌ 某些测试失败。请检查模型配置和API兼容性。")


if __name__ == "__main__":
    asyncio.run(main())