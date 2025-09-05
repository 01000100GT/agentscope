# -*- coding: utf-8 -*-
"""The main entry point of the Deep Research agent example."""
# 深度研究智能体示例的主入口点

import asyncio  # 导入异步IO库
import os  # 导入操作系统接口模块
from pathlib import Path  # 导入路径处理模块

# Load environment variables from .env file
# 从.env文件加载环境变量
try:
    from dotenv import load_dotenv  # 导入环境变量加载器

    # Look for .env file in project root
    # 在项目根目录中查找.env文件
    env_path = Path(__file__).parent.parent.parent / ".env"  # 构建.env文件路径
    print(f"🔍 尝试加载环境变量文件: {env_path}")
    result = load_dotenv(env_path)  # 加载环境变量
    print(f"   加载结果: {result}")
    
    # 验证环境变量是否加载成功
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if tavily_key:
        print(f"✅ TAVILY_API_KEY 已成功加载，长度: {len(tavily_key)} 字符")
    else:
        print("⚠️ TAVILY_API_KEY 未找到")
except ImportError:
    print(
        "python-dotenv not installed. Please install it with: pip install python-dotenv"
    )  # 提示未安装python-dotenv
    print("Or set environment variables manually.")  # 提示手动设置环境变量

from deep_research_agent import DeepResearchAgent  # 导入深度研究智能体

from agentscope import logger  # 导入日志记录器
from agentscope.formatter import OpenAIChatFormatter  # 导入OpenAI聊天格式化器
from agentscope.memory import InMemoryMemory  # 导入内存存储
from agentscope.model import OpenAIChatModel  # 导入OpenAI聊天模型
from agentscope.message import Msg  # 导入消息类
from agentscope.mcp import StdIOStatefulClient  # 导入标准输入输出有状态客户端


async def main(user_query: str) -> None:  # 定义主异步函数，接收用户查询参数
    """The main entry point for the Deep Research agent example."""
    # 深度研究智能体示例的主入口点
    logger.setLevel(level="DEBUG")  # 设置日志级别为DEBUG

    # 检查环境变量和依赖
    # 首先检查环境变量是否被正确加载
    print("🔍 检查环境变量加载情况:")
    tavily_api_key: str | None = os.environ.get("TAVILY_API_KEY")
    print(f"   TAVILY_API_KEY: {'已设置' if tavily_api_key else '未设置'}")
    if tavily_api_key:
        print(f"   API密钥长度: {len(tavily_api_key)} 字符")
        print(f"   API密钥前缀: {tavily_api_key[:10]}...")

    if not tavily_api_key:
        print("⚠️ 警告: 未设置 TAVILY_API_KEY，搜索功能可能受限")

    # 检查 npx 是否可用
    import shutil

    npx_path = shutil.which("npx")
    if not npx_path:
        print("❌ 错误: 未找到 npx 命令")
        print("💡 建议: 使用 main_no_search.py 运行不需要搜索工具的版本")
        print("或者确保 Node.js 已正确安装并在 PATH 中")
        return

    print(f"✅ 找到 npx: {npx_path}")

    tavily_search_client: StdIOStatefulClient = (
        StdIOStatefulClient(  # 创建Tavily搜索客户端
            name="tavily_mcp",  # 设置客户端名称
            command=npx_path,  # 使用完整路径
            args=["-y", "tavily-mcp@latest"],  # 设置命令参数
            env={"TAVILY_API_KEY": tavily_api_key or ""},  # 设置环境变量，包含API密钥
        )
    )

    default_working_dir = os.path.join(  # 构建默认工作目录路径
        os.path.dirname(__file__),  # 获取当前文件目录
        "deepresearch_agent_demo_env",  # 添加子目录名
    )
    agent_working_dir = os.getenv(  # 获取智能体工作目录
        "AGENT_OPERATION_DIR",  # 环境变量名
        default_working_dir,  # 默认值
    )
    os.makedirs(agent_working_dir, exist_ok=True)  # 创建工作目录，如果已存在则忽略

    try:
        # Get custom OpenAI API configuration
        # 获取自定义OpenAI API配置
        api_key = os.environ.get("CUSTOM_OPENAI_API_KEY")  # 获取API密钥
        base_url = os.environ.get("CUSTOM_OPENAI_BASE_URL")  # 获取基础URL
        model_name = os.environ.get(
            "CUSTOM_MODEL_NAME", "gpt-3.5-turbo"
        )  # 获取模型名称，默认为gpt-3.5-turbo

        if api_key is None:  # 检查API密钥是否存在
            raise ValueError(
                "CUSTOM_OPENAI_API_KEY environment variable is required"
            )  # 抛出错误：需要API密钥环境变量
        if base_url is None:  # 检查基础URL是否存在
            raise ValueError(
                "CUSTOM_OPENAI_BASE_URL environment variable is required"
            )  # 抛出错误：需要基础URL环境变量

        # 连接到Tavily搜索客户端
        print("🔗 正在连接到Tavily搜索客户端...")
        await tavily_search_client.connect()  # 连接到Tavily搜索客户端
        print("✅ Tavily搜索客户端连接成功")

        agent = DeepResearchAgent(  # 创建深度研究智能体实例
            name="Friday",  # 设置智能体名称
            sys_prompt="You are Friday, a helpful research assistant.",  # 简化系统提示
            model=OpenAIChatModel(  # 配置OpenAI聊天模型
                model_name=model_name,  # 设置模型名称
                api_key=api_key,  # 设置API密钥
                client_args={"base_url": base_url},  # 设置客户端参数
                stream=False,  # 关闭流式输出以提高稳定性
                generate_kwargs={  # 添加生成参数
                    "temperature": 0.7,  # 设置温度参数
                    "max_tokens": 2048,  # 减少最大令牌数以提高兼容性
                },
            ),
            formatter=OpenAIChatFormatter(),  # 设置聊天格式化器
            memory=InMemoryMemory(),  # 设置内存存储
            search_mcp_client=tavily_search_client,  # 设置搜索MCP客户端
            tmp_file_storage_dir=agent_working_dir,  # 设置临时文件存储目录
            max_iters=5,  # 减少最大迭代次数以提高稳定性
        )
        user_name = "Bob"  # 设置用户名
        msg = Msg(  # 创建消息对象
            user_name,  # 设置发送者
            content=user_query,  # 设置消息内容
            role="user",  # 设置角色为用户
        )
        print("🤖 智能体已初始化，开始处理查询...")
        result = await agent(msg)  # 调用智能体处理消息
        print("✅ 查询处理完成")
        print("📋 研究结果:")
        print("=" * 60)
        print(result.get_text_content())
        print("=" * 60)
        logger.info(result)  # 记录结果信息

    except Exception as err:  # 捕获异常
        print(f"❌ 执行过程中发生错误: {err}")
        import traceback
        traceback.print_exc()
        logger.exception(err)  # 记录异常信息
    finally:
        try:
            print("🔌 正在关闭Tavily搜索客户端...")
            await tavily_search_client.close()  # 关闭Tavily搜索客户端连接
            print("✅ Tavily搜索客户端已关闭")
        except Exception as e:
            print(f"⚠️ 关闭Tavily搜索客户端时发生错误: {e}")
            logger.warning(f"关闭Tavily搜索客户端时发生错误: {e}")


if __name__ == "__main__":  # 判断是否为主程序入口
    query = (  # 定义查询问题
        "如果埃利乌德·基普乔格能够无限期地保持他创纪录的"
        "马拉松配速，那么他跑完地球到月球最近距离"
        "需要多少千小时？请使用维基百科月球页面上的"
        "最小近地点值来进行计算。将结果四舍五入"
        "到最接近的1000小时，如有必要请不要使用"
        "任何逗号分隔符。"
    )
    try:
        asyncio.run(main(query))  # 运行主异步函数
    except Exception as e:  # 捕获异常
        print(f"❌ 程序执行过程中发生错误: {e}")
        logger.exception(e)  # 记录异常信息