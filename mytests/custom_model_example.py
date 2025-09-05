# -*- coding: utf-8 -*-
"""
自定义模型使用示例

演示如何在 AgentScope 中配置和使用各种自定义模型，
包括本地模型、自建 API、第三方兼容接口等。
"""

import os
import asyncio
from dotenv import load_dotenv
from typing import AsyncGenerator, Any, List, Literal
from datetime import datetime

import agentscope
from agentscope.agent import ReActAgent, UserAgent
from agentscope.formatter import (
    OpenAIChatFormatter,
    DashScopeChatFormatter,
    OllamaChatFormatter,
)
from agentscope.memory import InMemoryMemory
from agentscope.model import (
    OpenAIChatModel,
    DashScopeChatModel,
    OllamaChatModel,
    ChatModelBase,
    ChatResponse,
)
from agentscope.tool import Toolkit, execute_python_code
from agentscope.message import Msg

# 加载环境变量
load_dotenv()


# =============================================================================
# 自定义模型实现示例
# =============================================================================


class CustomOpenAICompatibleModel(ChatModelBase):
    """
    自定义 OpenAI 兼容模型类

    适用于：
    - vLLM 部署的模型
    - FastChat 部署的模型
    - LocalAI
    - 其他 OpenAI 兼容的 API
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> None:
        """初始化自定义 OpenAI 兼容模型

        Args:
            model_name: 模型名称
            api_key: API 密钥
            base_url: API 基础 URL
            stream: 是否使用流式输出
            temperature: 温度参数
            max_tokens: 最大输出 token 数
        """
        super().__init__(model_name, stream)

        try:
            import openai
        except ImportError as e:
            raise ImportError("请安装 openai 库: pip install openai") from e

        self.client = openai.AsyncClient(api_key=api_key, base_url=base_url, **kwargs)
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def __call__(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: Literal["auto", "none", "any", "required"] | str | None = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        """调用自定义模型"""

        # 准备请求参数
        request_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "stream": self.stream,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs,
        }

        if tools:
            request_kwargs["tools"] = tools

        if tool_choice:
            request_kwargs["tool_choice"] = tool_choice

        start_time = datetime.now()

        try:
            response = await self.client.chat.completions.create(**request_kwargs)
            
            if self.stream:
                return self._parse_stream_response(start_time, response)
            else:
                return self._parse_response(start_time, response)

        except Exception as e:
            # 简单的错误处理
            from agentscope.message import TextBlock

            return ChatResponse(
                content=[TextBlock(type="text", text=f"模型调用失败: {str(e)}")],
                usage=None,
            )

    def _parse_response(self, start_time: datetime, response) -> ChatResponse:
        """解析非流式响应"""
        from agentscope.message import TextBlock
        from agentscope.model._model_usage import ChatUsage

        content_blocks = []
        if response.choices and response.choices[0].message.content:
            content_blocks.append(
                TextBlock(type="text", text=response.choices[0].message.content)
            )

        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = ChatUsage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
                time=(datetime.now() - start_time).total_seconds(),
            )

        return ChatResponse(content=content_blocks, usage=usage)

    async def _parse_stream_response(
        self, start_time: datetime, response
    ) -> AsyncGenerator[ChatResponse, None]:
        """解析流式响应"""
        from agentscope.message import TextBlock
        from agentscope.model._model_usage import ChatUsage

        accumulated_text = ""

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                accumulated_text += content

                yield ChatResponse(
                    content=[TextBlock(type="text", text=accumulated_text)], usage=None
                )

        # 最终响应包含使用情况
        usage = ChatUsage(
            input_tokens=0,  # 流式响应通常不包含使用统计
            output_tokens=0,
            time=(datetime.now() - start_time).total_seconds(),
        )

        yield ChatResponse(
            content=[TextBlock(type="text", text=accumulated_text)], usage=usage
        )


# =============================================================================
# 配置工厂函数
# =============================================================================


def create_openai_model():
    """创建 OpenAI 模型"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请设置 OPENAI_API_KEY 环境变量")

    return OpenAIChatModel(
        model_name="gpt-4",
        api_key=api_key,
        stream=True,
    ), OpenAIChatFormatter()


def create_dashscope_model():
    """创建 DashScope 模型"""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")

    return DashScopeChatModel(
        model_name="qwen-max",
        api_key=api_key,
        stream=True,
    ), DashScopeChatFormatter()


def create_ollama_model():
    """创建 Ollama 本地模型"""
    return OllamaChatModel(
        model_name=os.environ.get("OLLAMA_MODEL_NAME", "llama3:latest"),
        host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        stream=True,
    ), OllamaChatFormatter()


def create_custom_openai_compatible_model():
    """创建自定义 OpenAI 兼容模型"""
    return CustomOpenAICompatibleModel(
        model_name=os.environ.get("CUSTOM_MODEL_NAME", "custom-model"),
        api_key=os.environ.get("CUSTOM_OPENAI_API_KEY", "dummy-key"),
        base_url=os.environ.get("CUSTOM_OPENAI_BASE_URL", "http://localhost:8000/v1"),
        temperature=float(os.environ.get("CUSTOM_MODEL_TEMPERATURE", "0.7")),
        max_tokens=int(os.environ.get("CUSTOM_MODEL_MAX_TOKENS", "2048")),
        stream=True,
    ), OpenAIChatFormatter()


def create_vllm_model():
    """创建 vLLM 部署的模型"""
    return CustomOpenAICompatibleModel(
        model_name="Qwen/Qwen2.5-7B-Instruct",  # vLLM 部署的模型名
        api_key="EMPTY",  # vLLM 通常不需要真实的 API key
        base_url="http://localhost:8000/v1",  # vLLM 服务地址
        temperature=0.7,
        max_tokens=2048,
        stream=True,
    ), OpenAIChatFormatter()


# =============================================================================
# 使用示例
# =============================================================================


async def test_custom_models():
    """测试各种自定义模型"""

    # 初始化 AgentScope
    agentscope.init(
        project=os.environ.get("PROJECT_NAME", "自定义模型测试"),
        name=os.environ.get("RUN_NAME", "测试运行"),
        studio_url=os.environ.get("STUDIO_URL"),
        logging_level=os.environ.get("LOGGING_LEVEL", "INFO"),
    )

    # 测试不同的模型配置
    model_configs = {
        "DashScope": create_dashscope_model,
        "OpenAI": create_openai_model,
        "Ollama": create_ollama_model,
        "自定义兼容模型": create_custom_openai_compatible_model,
        "vLLM模型": create_vllm_model,
    }

    for model_name, create_model_func in model_configs.items():
        print(f"\n{'=' * 50}")
        print(f"测试 {model_name} 模型")
        print(f"{'=' * 50}")

        try:
            model, formatter = create_model_func()

            # 为每个模型创建独立的工具包
            toolkit = Toolkit()
            toolkit.register_tool_function(execute_python_code)

            # 创建智能体
            agent = ReActAgent(
                name=f"{model_name}助手",
                sys_prompt=f"你是使用 {model_name} 模型的智能助手。",
                model=model,
                formatter=formatter,
                toolkit=toolkit,
                memory=InMemoryMemory(),
            )

            # 测试对话
            test_msg = Msg("用户", f"你好！请介绍一下你使用的是什么模型。", "user")

            response = await agent(test_msg)
            print(f"模型响应: {response.get_text_content()}")

        except Exception as e:
            print(f"测试 {model_name} 模型时出错: {str(e)}")
            continue


async def interactive_custom_model():
    """交互式自定义模型测试"""

    agentscope.init(
        project="交互式自定义模型测试",
        name="用户交互",
        studio_url=os.environ.get("STUDIO_URL"),
        logging_level="INFO",
    )

    # 让用户选择模型
    print("可用的模型配置:")
    print("1. DashScope (阿里云灵积)")
    print("2. OpenAI")
    print("3. Ollama (本地)")
    print("4. 自定义 OpenAI 兼容模型")
    print("5. vLLM 部署模型")

    choice = input("请选择模型 (1-5): ").strip()

    model_map = {
        "1": create_dashscope_model,
        "2": create_openai_model,
        "3": create_ollama_model,
        "4": create_custom_openai_compatible_model,
        "5": create_vllm_model,
    }

    if choice not in model_map:
        print("无效选择，使用默认 DashScope 模型")
        choice = "1"

    try:
        model, formatter = model_map[choice]()

        toolkit = Toolkit()
        toolkit.register_tool_function(execute_python_code)

        agent = ReActAgent(
            name="自定义模型助手",
            sys_prompt="你是一个有用的AI助手，可以回答问题和执行代码。",
            model=model,
            formatter=formatter,
            toolkit=toolkit,
            memory=InMemoryMemory(),
        )

        user = UserAgent(name="用户")

        print("\n开始对话 (输入 'exit' 退出):")
        msg = None
        while True:
            msg = await user(msg)
            content = msg.get_text_content()
            if content and content.lower() == "exit":
                break
            msg = await agent(msg)

    except Exception as e:
        print(f"初始化模型失败: {str(e)}")
        print("请检查 .env 文件中的配置是否正确")


if __name__ == "__main__":
    print("AgentScope 自定义模型配置示例")
    print("=" * 50)

    mode = input("选择运行模式 (1: 批量测试, 2: 交互式): ").strip()

    if mode == "1":
        asyncio.run(test_custom_models())
    else:
        asyncio.run(interactive_custom_model())
