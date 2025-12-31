"""LangGraph 工作流定义（简化版 - 无意图检测）"""

from typing import Annotated, Any, Dict, List, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .config import get_config
from .excel_loader import get_loader
from .language import detect_target_language, is_language_mismatch, language_label, localize, rewrite_system_prompt
from .prompts import SYSTEM_PROMPT
from .tools import ALL_TOOLS


class AgentState(TypedDict):
    """Agent 状态"""
    messages: Annotated[List[BaseMessage], add_messages]


def get_llm():
    """获取 LLM 实例"""
    config = get_config()
    provider = config.model.get_active_provider()
    return ChatOpenAI(
        model=provider.model_name,
        api_key=provider.api_key,
        base_url=provider.base_url if provider.base_url else None,
        temperature=provider.temperature,
        max_tokens=provider.max_tokens,
    )


def _last_user_text(messages: List[BaseMessage]) -> str:
    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage):
            return msg.content or ""
    return ""


def agent_node(state: AgentState) -> AgentState:
    """Agent 节点 - 规划和执行"""
    loader = get_loader()
    user_text = _last_user_text(state.get("messages", []))
    target_language = detect_target_language(user_text)

    excel_summary = (
        loader.get_summary(language=target_language)
        if loader.is_loaded
        else localize(
            target_language,
            en="No Excel file loaded.",
            zh="未加载 Excel 文件",
        )
    )

    # 构建系统消息
    system_message = SystemMessage(
        content=SYSTEM_PROMPT.format(
            excel_summary=excel_summary,
            target_language=language_label(target_language),
        )
    )

    # 获取带工具的 LLM
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # 构建消息列表
    messages = [system_message] + state["messages"]

    # 调用 LLM
    response = llm_with_tools.invoke(messages)

    # Final-answer language enforcement (never rewrite tool calls).
    if (
        isinstance(response, AIMessage)
        and not response.tool_calls
        and response.content
        and is_language_mismatch(target_language, response.content)
    ):
        rewriter_llm = get_llm()
        rewritten = rewriter_llm.invoke(
            [
                SystemMessage(content=rewrite_system_prompt(target_language)),
                HumanMessage(content=response.content),
            ]
        )
        if isinstance(rewritten, AIMessage) and rewritten.content:
            response = AIMessage(content=rewritten.content)

    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """判断是否需要继续调用工具"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "end"


def build_graph() -> StateGraph:
    """构建 LangGraph 工作流（简化版）"""
    
    # 创建工具节点
    tool_node = ToolNode(ALL_TOOLS)
    
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # 设置入口 - 直接进入 agent
    workflow.set_entry_point("agent")
    
    # 添加边
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        }
    )
    
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()


# 全局图实例
_graph = None


def get_graph():
    """获取图实例"""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def reset_graph():
    """重置图实例"""
    global _graph
    _graph = None
