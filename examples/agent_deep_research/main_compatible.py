# -*- coding: utf-8 -*-
"""兼容SiliconFlow API的深度研究智能体示例"""

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


class CompatibleDeepResearchAgent(ReActAgent):
    """兼容SiliconFlow API的简化深度研究智能体"""
    
    def __init__(self, *args, search_mcp_client=None, **kwargs):
        """初始化兼容的智能体"""
        # 使用更简单但准确的系统提示符
        simple_sys_prompt = """You are Friday, a helpful research assistant. Your capabilities include:

1. Conducting thorough research using search tools
2. Analyzing and synthesizing information from multiple sources  
3. Providing accurate, well-structured responses
4. Citing sources when applicable

When conducting research:
- Search for current, authoritative information
- Cross-reference multiple sources
- Present information clearly and logically
- Be precise about dates, names, and facts

Always strive for accuracy and completeness in your research."""
        
        # 替换复杂的系统提示符
        if 'sys_prompt' in kwargs:
            kwargs['sys_prompt'] = simple_sys_prompt
        
        super().__init__(*args, **kwargs)
        
        # 注册搜索工具
        if search_mcp_client:
            asyncio.create_task(self._register_search_tools(search_mcp_client))
    
    async def _register_search_tools(self, search_client):
        """注册搜索工具"""
        try:
            await search_client.connect()
            await self.toolkit.register_mcp_client(search_client)
            print("✅ 搜索工具注册成功")
        except Exception as e:
            print(f"⚠️ 搜索工具注册失败: {e}")


async def main_compatible(user_query: str) -> None:
    """兼容版本的主函数"""
    print(f"🚀 开始执行兼容版本的深度研究...")
    print(f"📝 查询: {user_query}")
    print("=" * 60)

    # 创建搜索客户端
    tavily_search_client = StdIOStatefulClient(
        name="tavily_mcp",
        command="npx",
        args=["-y", "tavily-mcp@latest"],
        env={"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "")},
    )

    # 获取自定义模型配置
    api_key = os.environ.get("CUSTOM_OPENAI_API_KEY")
    base_url = os.environ.get("CUSTOM_OPENAI_BASE_URL")
    model_name = os.environ.get("CUSTOM_MODEL_NAME", "gpt-3.5-turbo")

    if not api_key or not base_url:
        raise ValueError("需要设置 CUSTOM_OPENAI_API_KEY 和 CUSTOM_OPENAI_BASE_URL")

    try:
        print("🤖 创建兼容的研究智能体...")
        
        # 创建工具包
        toolkit = Toolkit()
        
        # 创建兼容的智能体
        agent = CompatibleDeepResearchAgent(
            name="Friday",
            sys_prompt="You are Friday, a helpful research assistant.",  # 简化的提示符
            model=OpenAIChatModel(
                model_name=model_name,
                api_key=api_key,
                client_args={"base_url": base_url},
                stream=False,  # 关闭流式输出
                generate_kwargs={
                    "temperature": 0.7,
                    "max_tokens": 2048,  # 减少最大令牌数
                },
            ),
            formatter=OpenAIChatFormatter(),
            memory=InMemoryMemory(),
            toolkit=toolkit,
            search_mcp_client=tavily_search_client,
        )
        
        print("✅ 智能体创建成功")
        print("🔍 开始研究...")
        
        # 创建用户消息
        msg = Msg("user", user_query, "user")
        
        # 执行研究
        result = await agent(msg)
        
        print("\n" + "=" * 60)
        print("🎉 研究完成!")
        print("📋 结果:")
        print(result.get_text_content())
        print("=" * 60)

    except Exception as err:
        print(f"❌ 执行错误: {err}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            await tavily_search_client.close()
            print("✅ 清理完成")
        except:
            pass


async def test_simple_query():
    """测试简单查询"""
    print("🧪 测试简单查询（不需要搜索）...")
    
    api_key = os.environ.get("CUSTOM_OPENAI_API_KEY")
    base_url = os.environ.get("CUSTOM_OPENAI_BASE_URL")
    model_name = os.environ.get("CUSTOM_MODEL_NAME", "gpt-3.5-turbo")

    if not api_key or not base_url:
        print("❌ 环境变量未设置")
        return

    try:
        # 创建简单的智能体（无搜索工具）
        agent = ReActAgent(
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
            toolkit=Toolkit(),  # 空工具包
        )
        
        # 测试简单问题
        msg = Msg("user", "请简要介绍人工智能", "user")
        result = await agent(msg)
        
        print("✅ 简单查询测试成功")
        print(f"响应: {result.get_text_content()[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ 简单查询测试失败: {e}")
        return False


if __name__ == "__main__":
    print("选择执行模式:")
    print("1. 测试简单查询（无搜索工具）")
    print("2. 兼容版本深度研究（带搜索工具）")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        try:
            success = asyncio.run(test_simple_query())
            if success:
                print("\n💡 简单查询成功，现在可以尝试选项2")
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断")
    else:
        query = "请简要介绍人工智能的发展历史"
        try:
            asyncio.run(main_compatible(query))
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断")
        except Exception as e:
            print(f"\n❌ 执行失败: {e}")