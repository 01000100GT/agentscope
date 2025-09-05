# -*- coding: utf-8 -*-
"""改进版本的深度研究智能体示例 - 针对SiliconFlow API优化"""

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
from agentscope.agent import ReActAgent
from agentscope.tool import Toolkit
from agentscope.mcp import StdIOStatefulClient


class ImprovedResearchAgent(ReActAgent):
    """改进的研究智能体 - 专为SiliconFlow API优化"""

    def __init__(self, *args, search_mcp_client=None, **kwargs):
        """初始化优化的研究智能体"""

        # 使用专门优化的系统提示符
        optimized_sys_prompt = """You are Friday, an advanced research assistant. Your core capabilities:

**Research Skills:**
• Search current information using available tools
• Analyze and synthesize data from multiple sources
• Provide structured, accurate responses
• Verify facts and cross-reference information

**Response Guidelines:**
• Be precise and factual
• Structure information logically
• Include relevant details and context
• Cite sources when applicable

**Tool Usage:**
• Use tavily-search for finding information
• Use tavily-extract for detailed content analysis
• Keep searches focused and relevant

Always prioritize accuracy and completeness in your research."""

        # 替换系统提示符
        if "sys_prompt" in kwargs:
            kwargs["sys_prompt"] = optimized_sys_prompt

        super().__init__(*args, **kwargs)

        # 注册搜索工具
        if search_mcp_client:
            asyncio.create_task(self._register_search_tools(search_mcp_client))

    async def _register_search_tools(self, search_client):
        """注册搜索工具"""
        try:
            await search_client.connect()
            await self.toolkit.register_mcp_client(search_client)
            logger.info("✅ 搜索工具注册成功")
        except Exception as e:
            logger.warning(f"⚠️ 搜索工具注册失败: {e}")


async def main_improved(user_query: str) -> None:
    """改进版本的主函数"""
    print("🚀 启动改进版深度研究智能体...")
    print(f"📝 研究主题: {user_query}")
    print("=" * 60)

    # 环境变量检查
    tavily_api_key = os.getenv("TAVILY_API_KEY", "")
    if not tavily_api_key:
        print("⚠️ 警告: 未设置 TAVILY_API_KEY，搜索功能可能受限")

    # 创建搜索客户端
    tavily_search_client = StdIOStatefulClient(
        name="tavily_mcp",
        command="npx",
        args=["-y", "tavily-mcp@latest"],
        env={"TAVILY_API_KEY": tavily_api_key},
    )

    # 获取自定义模型配置
    api_key = os.environ.get("CUSTOM_OPENAI_API_KEY")
    base_url = os.environ.get("CUSTOM_OPENAI_BASE_URL")
    model_name = os.environ.get(
        "CUSTOM_MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    )

    if not api_key or not base_url:
        raise ValueError("需要设置 CUSTOM_OPENAI_API_KEY 和 CUSTOM_OPENAI_BASE_URL")

    try:
        print("🤖 初始化改进版研究智能体...")
        print(f"   🎯 模型: {model_name}")
        print(f"   🌐 API: {base_url}")

        # 创建优化的智能体
        agent = ImprovedResearchAgent(
            name="Friday",
            sys_prompt="Research assistant prompt will be set by the class",
            model=OpenAIChatModel(
                model_name=model_name,
                api_key=api_key,
                client_args={"base_url": base_url},
                stream=False,  # 关闭流式输出以提高稳定性
                generate_kwargs={
                    "temperature": 0.3,  # 降低温度以提高准确性
                    "max_tokens": 3000,
                    "top_p": 0.9,
                },
            ),
            formatter=OpenAIChatFormatter(),
            memory=InMemoryMemory(),
            toolkit=Toolkit(),
            search_mcp_client=tavily_search_client,
            max_iters=8,  # 适中的迭代次数
        )

        print("✅ 智能体初始化完成")
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


async def interactive_mode():
    """交互模式"""
    print("🔄 进入交互模式")
    print("💡 您可以输入任何研究主题，输入 'quit' 退出")

    while True:
        print("\n" + "-" * 40)
        query = input("📝 请输入研究主题: ").strip()

        if query.lower() in ["quit", "exit", "退出"]:
            print("👋 再见!")
            break

        if query:
            try:
                await main_improved(query)
            except KeyboardInterrupt:
                print("\n⚠️ 研究被中断")
            except Exception as e:
                print(f"❌ 研究失败: {e}")


if __name__ == "__main__":
    # 预定义的研究主题
    sample_topics = {
        "1": "人工智能的发展历史",
        "2": "区块链技术的应用前景",
        "3": "可再生能源的最新进展",
        "4": "量子计算的原理与挑战",
        "5": "互动模式 - 自定义研究主题",
    }

    print("🎯 选择研究主题:")
    for key, topic in sample_topics.items():
        print(f"   {key}. {topic}")

    choice = input("\n请选择 (1-5): ").strip()

    if choice in ["1", "2", "3", "4"]:
        topic = sample_topics[choice]
        print(f"\n🎯 选择的研究主题: {topic}")
        try:
            asyncio.run(main_improved(topic))
        except KeyboardInterrupt:
            print("\n⚠️ 研究被中断")
    elif choice == "5":
        try:
            asyncio.run(interactive_mode())
        except KeyboardInterrupt:
            print("\n⚠️ 退出交互模式")
    else:
        print("❌ 无效选择")
