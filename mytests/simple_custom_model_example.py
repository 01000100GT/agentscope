# -*- coding: utf-8 -*-
"""
简单的自定义模型使用示例

展示如何在 AgentScope 中使用 .env 文件配置的自定义模型
"""

import os
import asyncio
from dotenv import load_dotenv

import agentscope
from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel, DashScopeChatModel, OllamaChatModel
from agentscope.formatter import (
    OpenAIChatFormatter,
    DashScopeChatFormatter,
    OllamaChatFormatter,
)
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg

# 加载环境变量
load_dotenv()


async def main():
    """主函数"""

    # 从环境变量初始化 AgentScope
    agentscope.init(
        project=os.environ.get("PROJECT_NAME", "自定义模型项目"),
        name=os.environ.get("RUN_NAME", "测试运行"),
        studio_url=os.environ.get("STUDIO_URL"),
        logging_level=os.environ.get("LOGGING_LEVEL", "INFO"),
    )

    print("可用的模型配置:")
    print("1. DashScope (从 .env 读取)")
    print("2. OpenAI (从 .env 读取)")
    print("3. Ollama 本地模型 (从 .env 读取)")
    print("4. 自定义 OpenAI 兼容模型 (从 .env 读取)")

    choice = input("请选择模型 (1-4): ").strip()

    # 根据选择创建模型和格式化器
    if choice == "1":
        # DashScope 配置
        dashscope_key = os.environ.get("DASHSCOPE_API_KEY")
        if not dashscope_key:
            raise ValueError("DASHSCOPE_API_KEY 环境变量未设置")

        model = DashScopeChatModel(
            model_name="qwen-max",
            api_key=dashscope_key,
            stream=True,
        )
        formatter = DashScopeChatFormatter()
        model_name = "DashScope"

    elif choice == "2":
        # OpenAI 配置
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY 环境变量未设置")

        # 构建参数字典
        openai_args = {
            "model_name": "gpt-4",
            "api_key": openai_key,
            "stream": True,
        }

        # 只有当 organization 存在时才添加该参数
        org_id = os.environ.get("OPENAI_ORG_ID")
        if org_id:
            openai_args["organization"] = org_id

        model = OpenAIChatModel(**openai_args)
        formatter = OpenAIChatFormatter()
        model_name = "OpenAI"

    elif choice == "3":
        # Ollama 本地模型配置
        model = OllamaChatModel(
            model_name=os.environ.get("OLLAMA_MODEL_NAME", "llama3:latest"),
            host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            stream=True,
        )
        formatter = OllamaChatFormatter()
        model_name = "Ollama"

    elif choice == "4":
        # 自定义 OpenAI 兼容模型 (如 vLLM, FastChat 等)
        model = OpenAIChatModel(
            model_name=os.environ.get("CUSTOM_MODEL_NAME", "custom-model"),
            api_key=os.environ.get("CUSTOM_OPENAI_API_KEY", "EMPTY"),
            client_args={
                "base_url": os.environ.get(
                    "CUSTOM_OPENAI_BASE_URL", "http://localhost:8000/v1"
                )
            },
            stream=True,
        )
        formatter = OpenAIChatFormatter()
        model_name = "自定义模型"

    else:
        print("无效选择，使用默认 DashScope")
        dashscope_key = os.environ.get("DASHSCOPE_API_KEY")
        if not dashscope_key:
            raise ValueError("DASHSCOPE_API_KEY 环境变量未设置")

        model = DashScopeChatModel(
            model_name="qwen-max",
            api_key=dashscope_key,
            stream=True,
        )
        formatter = DashScopeChatFormatter()
        model_name = "DashScope (默认)"

    try:
        # 创建智能体
        agent = ReActAgent(
            name=f"{model_name}助手",
            sys_prompt=f"你是使用 {model_name} 的智能助手。请简要介绍你自己和你的能力。",
            model=model,
            formatter=formatter,
            memory=InMemoryMemory(),
        )

        # 测试消息
        test_msg = Msg("用户", "你好！请介绍一下你自己。", "user")

        print(f"\n使用 {model_name} 模型进行对话...")
        print("-" * 50)

        response = await agent(test_msg)
        print(f"模型响应: {response.get_text_content()}")

        print("\n✅ 模型配置成功！")
        print(
            "你可以在 Studio 中查看这次对话：",
            os.environ.get("STUDIO_URL", "Studio未配置"),
        )

    except Exception as e:
        print(f"❌ 模型配置失败: {str(e)}")
        print("\n请检查:")
        print("1. .env 文件中对应的 API 密钥是否正确设置")
        print("2. 对于本地模型 (如 Ollama)，确保服务正在运行")
        print("3. 对于自定义模型，确保服务器地址和端口正确")


if __name__ == "__main__":
    asyncio.run(main())
