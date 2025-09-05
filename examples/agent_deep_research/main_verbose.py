# -*- coding: utf-8 -*-
"""带详细日志的深度研究智能体示例"""

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
    print("Or set environment variables manually.")

from deep_research_agent import DeepResearchAgent

from agentscope import logger, setup_logger
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.message import Msg
from agentscope.mcp import StdIOStatefulClient


async def main_verbose(user_query: str) -> None:
    """带详细日志的主函数"""
    # 设置更详细的日志
    setup_logger(level="INFO")
    logger.setLevel("INFO")
    
    print(f"🚀 开始执行深度研究任务...")
    print(f"📝 查询问题: {user_query}")
    print("=" * 60)

    tavily_search_client = StdIOStatefulClient(
        name="tavily_mcp",
        command="npx",
        args=["-y", "tavily-mcp@latest"],
        env={"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "")},
    )

    default_working_dir = os.path.join(
        os.path.dirname(__file__),
        "deepresearch_agent_demo_env",
    )
    agent_working_dir = os.getenv(
        "AGENT_OPERATION_DIR",
        default_working_dir,
    )
    os.makedirs(agent_working_dir, exist_ok=True)

    try:
        print("🔌 正在连接到 Tavily 搜索服务...")
        await tavily_search_client.connect()
        print("✅ Tavily 搜索服务连接成功")

        # 获取自定义模型配置
        api_key = os.environ.get("CUSTOM_OPENAI_API_KEY")
        base_url = os.environ.get("CUSTOM_OPENAI_BASE_URL")
        model_name = os.environ.get("CUSTOM_MODEL_NAME", "gpt-3.5-turbo")

        if api_key is None:
            raise ValueError("CUSTOM_OPENAI_API_KEY environment variable is required")
        if base_url is None:
            raise ValueError("CUSTOM_OPENAI_BASE_URL environment variable is required")

        print(f"🤖 正在初始化智能体...")
        print(f"   模型: {model_name}")
        print(f"   API: {base_url}")
        
        agent = DeepResearchAgent(
            name="Friday",
            sys_prompt="You are a helpful assistant named Friday.",
            model=OpenAIChatModel(
                model_name=model_name,
                api_key=api_key,
                client_args={"base_url": base_url},
                stream=False,
                generate_kwargs={
                    "temperature": 0.7,
                    "max_tokens": 4096,
                },
            ),
            formatter=OpenAIChatFormatter(),
            memory=InMemoryMemory(),
            search_mcp_client=tavily_search_client,
            tmp_file_storage_dir=agent_working_dir,
        )
        
        print("✅ 智能体初始化完成")
        print("🔍 开始执行深度研究...")
        print("⏳ 这可能需要几分钟时间，请耐心等待...")
        print("=" * 60)
        
        user_name = "Bob"
        msg = Msg(
            user_name,
            content=user_query,
            role="user",
        )
        
        print("📤 发送查询到智能体...")
        result = await agent(msg)
        
        print("\n" + "=" * 60)
        print("🎉 研究完成!")
        print("📋 结果:")
        print(result.get_text_content())
        print("=" * 60)
        
        logger.info(result)

    except Exception as err:
        print(f"❌ 执行过程中出现错误: {err}")
        logger.exception(err)
    finally:
        print("🔌 正在关闭 Tavily 搜索服务...")
        await tavily_search_client.close()
        print("✅ 清理完成")


if __name__ == "__main__":
    # 使用一个简单一些的测试问题
    simple_query = "请简要介绍一下人工智能的发展历史"
    
    print("选择执行模式:")
    print("1. 简单测试查询")
    print("2. 原始复杂查询（马拉松计算问题）")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        query = simple_query
    else:
        query = (
            "如果埃利乌德·基普乔格能够无限期地保持他创纪录的"
            "马拉松配速，那么他跑完地球到月球最近距离"
            "需要多少千小时？请使用维基百科月球页面上的"
            "最小近地点值来进行计算。将结果四舍五入"
            "到最接近的1000小时，如有必要请不要使用"
            "任何逗号分隔符。"
        )
    
    try:
        asyncio.run(main_verbose(query))
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断了执行")
    except Exception as e:
        print(f"\n\n❌ 程序执行失败: {e}")