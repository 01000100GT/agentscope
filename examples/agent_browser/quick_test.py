#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速测试自定义模型的脚本"""

import asyncio
import os
import sys
from typing import Union, AsyncGenerator

# 直接设置环境变量，不依赖 dotenv
os.environ["CUSTOM_OPENAI_API_KEY"] = "sk-sbwxkcjoeolowiokiusntqcqjbwqtyqlzrlxvjfykpspliqd"
os.environ["CUSTOM_OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"
os.environ["CUSTOM_MODEL_NAME"] = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

from agentscope.model import OpenAIChatModel
from agentscope.model._model_response import ChatResponse
from agentscope.message import Msg

async def quick_test():
    """快速测试模型"""
    print("🚀 快速测试自定义模型...")
    
    try:
        model = OpenAIChatModel(
            model_name=os.environ["CUSTOM_MODEL_NAME"],
            api_key=os.environ["CUSTOM_OPENAI_API_KEY"],
            client_args={"base_url": os.environ["CUSTOM_OPENAI_BASE_URL"]},
            stream=False,  # 关闭流式输出以简化测试
        )
        
        # 简单测试消息
        messages = [{"role": "user", "content": "你好，请简单回答：1+1等于几？"}]
        
        response: Union[ChatResponse, AsyncGenerator[ChatResponse, None]] = await model(messages)
        
        # 由于stream=False，这里应该返回ChatResponse而不是AsyncGenerator
        if isinstance(response, ChatResponse):
            content_text = "无响应内容"
            if response.content:
                for block in response.content:
                    block_dict = dict(block)
                    if block_dict.get("type") == "text":
                        content_text = block_dict.get("text", "")
                        break
                    elif block_dict.get("type") == "thinking":
                        content_text = block_dict.get("thinking", "")
                        break
                    elif block_dict.get("type") == "tool_use":
                        content_text = f"工具调用: {block_dict.get('name', '')} - {block_dict.get('input', {})}"
                        break
        else:
            content_text = "流式响应不支持在此测试中"
                    
        print(f"✅ 模型响应: {content_text}")
        print("✅ 自定义模型配置成功！")
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(quick_test())
    sys.exit(0 if success else 1)