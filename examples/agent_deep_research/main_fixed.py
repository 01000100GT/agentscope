# -*- coding: utf-8 -*-
"""修复版本的深度研究智能体示例"""

import asyncio
import os
from pathlib import Path

# 加载环境变量
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    print("python-dotenv not installed.")

from agentscope import logger
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.message import Msg
from agentscope.mcp import StdIOStatefulClient
from deep_research_agent import DeepResearchAgent


async def main_fixed(user_query: str) -> None:
    """修复版本的主函数"""
    print("🚀 启动修复版深度研究智能体...")
    print(f"📝 研究主题: {user_query}")
    print("=" * 60)

    # 环境变量检查
    tavily_api_key = os.getenv("TAVILY_API_KEY", "")
    if not tavily_api_key:
        print("⚠️ 警告: 未设置 TAVILY_API_KEY，搜索功能可能受限")

    # 创建搜索客户端
    try:
        import shutil
        npx_path = shutil.which("npx")
        if not npx_path:
            print("❌ 错误: 未找到 npx 命令")
            return
            
        tavily_search_client = StdIOStatefulClient(
            name="tavily_mcp",
            command=npx_path,
            args=["-y", "tavily-mcp@latest"],
            env={"TAVILY_API_KEY": tavily_api_key},
        )
    except Exception as e:
        print(f"❌ 创建Tavily搜索客户端失败: {e}")
        return

    # 获取自定义模型配置
    api_key = os.environ.get("CUSTOM_OPENAI_API_KEY")
    base_url = os.environ.get("CUSTOM_OPENAI_BASE_URL")
    model_name = os.environ.get(
        "CUSTOM_MODEL_NAME", "gpt-3.5-turbo"
    )

    if not api_key or not base_url:
        raise ValueError("需要设置 CUSTOM_OPENAI_API_KEY 和 CUSTOM_OPENAI_BASE_URL")

    try:
        print("🤖 初始化修复版研究智能体...")
        print(f"   🎯 模型: {model_name}")
        print(f"   🌐 API: {base_url}")

        # 连接到Tavily搜索客户端
        print("🔗 正在连接到Tavily搜索客户端...")
        await tavily_search_client.connect()
        print("✅ Tavily搜索客户端连接成功")

        # 创建修复的智能体
        agent = DeepResearchAgent(
            name="Friday",
            sys_prompt="You are Friday, a helpful research assistant.",
            model=OpenAIChatModel(
                model_name=model_name,
                api_key=api_key,
                client_args={"base_url": base_url},
                stream=False,  # 关闭流式输出以提高稳定性
                generate_kwargs={
                    "temperature": 0.7,  # 适中的温度
                    "max_tokens": 2048,
                },
            ),
            formatter=OpenAIChatFormatter(),
            memory=InMemoryMemory(),
            search_mcp_client=tavily_search_client,
            max_iters=5,  # 减少迭代次数
        )

        # 等待工具注册完成
        print("⏳ 等待工具注册完成...")
        await asyncio.sleep(2)  # 给工具注册一些时间
        print("✅ 工具注册完成")

        print("🔍 开始深度研究...")
        print("⏳ 请稍等，正在搜索和分析相关信息...")
        print("=" * 60)

        # 创建研究请求
        msg = Msg("user", user_query, "user")

        # 执行研究
        result = await agent(msg)

        print("\n" + "=" * 60)
        print("🎉 研究完成!")
        print("📋 研究报告:")
        print("=" * 60)
        print(result.get_text_content())
        print("=" * 60)

    except Exception as err:
        print(f"❌ 执行过程中发生错误: {err}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            await tavily_search_client.close()
            print("\n🔌 清理完成")
        except Exception:
            # 忽略清理过程中的异常，但不阻止程序退出
            pass


if __name__ == "__main__":
    query = (
        "如果埃利乌德·基普乔格能够无限期地保持他创纪录的"
        "马拉松配速，那么他跑完地球到月球最近距离"
        "需要多少千小时？请使用维基百科月球页面上的"
        "最小近地点值来进行计算。将结果四舍五入"
        "到最接近的1000小时，如有必要请不要使用"
        "任何逗号分隔符。"
    )
    
    try:
        asyncio.run(main_fixed(query))
    except KeyboardInterrupt:
        print("\n⚠️ 研究被中断")
    except Exception as e:
        print(f"❌ 程序执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()