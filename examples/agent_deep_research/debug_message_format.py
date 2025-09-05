#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断消息格式问题的脚本"""

import asyncio
import os
from pathlib import Path

# 加载环境变量
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    print("python-dotenv not installed. Please install it with: pip install python-dotenv")

from deep_research_agent import DeepResearchAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.message import Msg
from agentscope.mcp import StdIOStatefulClient
import json


async def debug_message_format():
    """调试消息格式问题"""
    print("🔍 开始诊断消息格式问题...")
    
    # 创建一个模拟的搜索客户端（不实际连接）
    tavily_search_client = StdIOStatefulClient(
        name="tavily_mcp",
        command="echo",  # 使用echo命令避免实际连接
        args=["test"],
        env={"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "")},
    )

    # 获取自定义模型配置
    api_key = os.environ.get("CUSTOM_OPENAI_API_KEY")
    base_url = os.environ.get("CUSTOM_OPENAI_BASE_URL")
    model_name = os.environ.get("CUSTOM_MODEL_NAME", "gpt-3.5-turbo")

    if not api_key or not base_url:
        print("❌ 环境变量未设置")
        return

    try:
        print("🤖 创建 DeepResearchAgent...")
        
        # 创建智能体但不连接搜索服务
        agent = DeepResearchAgent(
            name="Friday",
            sys_prompt="You are a helpful assistant named Friday.",
            model=OpenAIChatModel(
                model_name=model_name,
                api_key=api_key,
                client_args={"base_url": base_url},
                stream=False,
            ),
            formatter=OpenAIChatFormatter(),
            memory=InMemoryMemory(),
            search_mcp_client=tavily_search_client,  # 不会实际连接
        )

        print("✅ 智能体创建成功")
        
        # 创建测试消息
        test_msg = Msg("user", "你好", "user")
        
        # 将消息添加到记忆中
        await agent.memory.add(test_msg)
        
        print("📝 检查系统提示符:")
        print(f"系统提示符长度: {len(agent.sys_prompt)} 字符")
        print("系统提示符前500字符:")
        print("-" * 60)
        print(agent.sys_prompt[:500])
        print("-" * 60)
        
        # 获取格式化后的消息
        print("\n🔍 检查格式化后的消息:")
        
        # 模拟 _reasoning 方法中的消息格式化过程
        msgs = [
            Msg("system", agent.sys_prompt, "system"),
            *await agent.memory.get_memory(),
        ]
        
        print(f"消息总数: {len(msgs)}")
        for i, msg in enumerate(msgs):
            print(f"消息 {i+1}: role={msg.role}, name={msg.name}, content_type={type(msg.content)}")
        
        # 使用格式化器格式化消息
        formatted_msgs = await agent.formatter.format(msgs)
        
        print(f"\n📤 格式化后的消息结构:")
        print(f"消息数量: {len(formatted_msgs)}")
        
        for i, msg in enumerate(formatted_msgs):
            print(f"\n消息 {i+1}:")
            print(f"  role: {msg.get('role')}")
            print(f"  name: {msg.get('name', 'N/A')}")
            print(f"  content: {type(msg.get('content'))}")
            if isinstance(msg.get('content'), list) and len(msg['content']) > 0:
                print(f"  content[0]: {msg['content'][0]}")
        
        # 保存详细信息到文件
        debug_info = {
            "system_prompt": agent.sys_prompt,
            "formatted_messages": formatted_msgs
        }
        
        with open("debug_message_format.json", "w", encoding="utf-8") as f:
            json.dump(debug_info, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 详细信息已保存到 debug_message_format.json")
        
    except Exception as e:
        print(f"❌ 调试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_message_format())