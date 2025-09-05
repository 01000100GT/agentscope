# -*- coding: utf-8 -*-
"""
测试自定义 OpenAI 兼容模型
只测试你已配置的自定义模型
"""

import os
import asyncio
from dotenv import load_dotenv

import agentscope
from agentscope.agent import ReActAgent, UserAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.tool import Toolkit, execute_python_code
from agentscope.message import Msg

# 导入自定义模型类
from mytests.custom_model_example import CustomOpenAICompatibleModel

# 加载环境变量
load_dotenv()


def create_custom_model():
    """创建自定义 OpenAI 兼容模型"""
    api_key = os.environ.get("CUSTOM_OPENAI_API_KEY")
    base_url = os.environ.get("CUSTOM_OPENAI_BASE_URL")
    model_name = os.environ.get("CUSTOM_MODEL_NAME")
    
    if not api_key:
        raise ValueError("请在 .env 文件中设置 CUSTOM_OPENAI_API_KEY")
    if not base_url:
        raise ValueError("请在 .env 文件中设置 CUSTOM_OPENAI_BASE_URL")
    if not model_name:
        raise ValueError("请在 .env 文件中设置 CUSTOM_MODEL_NAME")
    
    print(f"🔧 配置自定义模型:")
    print(f"   模型名称: {model_name}")
    print(f"   服务地址: {base_url}")
    print(f"   API Key: {api_key[:20]}...")
    
    model = CustomOpenAICompatibleModel(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=float(os.environ.get("CUSTOM_MODEL_TEMPERATURE", "0.7")),
        max_tokens=int(os.environ.get("CUSTOM_MODEL_MAX_TOKENS", "2048")),
        stream=False,  # 暂时使用非流式模式
    )
    
    formatter = OpenAIChatFormatter()
    
    return model, formatter


async def test_custom_model():
    """测试自定义模型"""
    
    # 初始化 AgentScope（不连接 Studio）
    agentscope.init(
        project=os.environ.get("PROJECT_NAME", "自定义模型测试"),
        name=os.environ.get("RUN_NAME", "自定义模型运行"),
        studio_url=os.environ.get("STUDIO_URL"),  # 暂时不连接 Studio
        logging_level=os.environ.get("LOGGING_LEVEL", "INFO")
    )
    
    print("🚀 开始测试自定义 OpenAI 兼容模型...")
    print("=" * 60)
    
    try:
        # 创建模型
        model, formatter = create_custom_model()
        
        # 创建工具包
        toolkit = Toolkit()
        toolkit.register_tool_function(execute_python_code)
        
        # 创建智能体
        agent = ReActAgent(
            name="自定义模型助手",
            sys_prompt="你是一个有用的AI助手。请简要介绍你自己，并说明你可以帮助用户做什么。",
            model=model,
            formatter=formatter,
            toolkit=toolkit,
            memory=InMemoryMemory(),
        )
        
        # 测试消息
        test_msg = Msg(
            "用户", 
            "你好！请介绍一下你自己，并告诉我你可以做什么。", 
            "user"
        )
        
        print("💬 发送测试消息...")
        response = await agent(test_msg)
        
        print("✅ 模型响应:")
        print("-" * 40)
        print(response.get_text_content())
        print("-" * 40)
        
        print("\n🎯 测试成功！你可以在 Studio 中查看详细信息:")
        print(f"   {os.environ.get('STUDIO_FRONTEND_URL', 'Studio未配置')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        print("\n🔍 请检查:")
        print("1. .env 文件中的 CUSTOM_OPENAI_API_KEY 是否正确")
        print("2. .env 文件中的 CUSTOM_OPENAI_BASE_URL 是否可访问") 
        print("3. .env 文件中的 CUSTOM_MODEL_NAME 是否正确")
        print("4. 网络连接是否正常")
        return False


async def interactive_chat():
    """交互式对话"""
    
    agentscope.init(
        project="自定义模型交互测试",
        name="交互对话",
        studio_url=os.environ.get("STUDIO_URL"),  # 暂时不连接 Studio
        logging_level="INFO"
    )
    
    try:
        model, formatter = create_custom_model()
        
        toolkit = Toolkit()
        toolkit.register_tool_function(execute_python_code)
        
        agent = ReActAgent(
            name="自定义模型助手",
            sys_prompt="你是一个有用的AI助手，可以回答问题和执行Python代码。",
            model=model,
            formatter=formatter,
            toolkit=toolkit,
            memory=InMemoryMemory(),
        )
        
        user = UserAgent(name="用户")
        
        print("\n💬 开始与自定义模型对话 (输入 'exit' 退出):")
        print("=" * 50)
        
        msg = None
        while True:
            msg = await user(msg)
            content = msg.get_text_content()
            if content and content.lower() == "exit":
                print("👋 对话结束！")
                break
            msg = await agent(msg)
    
    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")


if __name__ == "__main__":
    print("🤖 AgentScope 自定义模型测试工具")
    print("=" * 50)
    
    # 检查必要的环境变量
    required_vars = ["CUSTOM_OPENAI_API_KEY", "CUSTOM_OPENAI_BASE_URL", "CUSTOM_MODEL_NAME"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"❌ 缺少必要的环境变量: {', '.join(missing_vars)}")
        print("请在 .env 文件中配置这些变量。")
        exit(1)
    
    mode = input("选择模式 (1: 快速测试, 2: 交互对话): ").strip()
    
    if mode == "1":
        result = asyncio.run(test_custom_model())
        if result:
            print("\n🎉 自定义模型配置成功！")
        else:
            print("\n💡 请根据上面的提示检查配置。")
    else:
        asyncio.run(interactive_chat())