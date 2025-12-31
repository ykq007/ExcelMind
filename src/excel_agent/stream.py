"""流式对话 - 使用手动工具解析避免 LangChain 工具绑定兼容性问题"""

import json
import re
from typing import Any, AsyncGenerator, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .domain.ports import LLMFactory, ToolRegistry
from .excel_loader import get_loader
from .knowledge_base import get_knowledge_base, format_knowledge_context
from .language import detect_target_language, is_language_mismatch, language_label, localize, rewrite_system_prompt
from .tools import ALL_TOOLS
from .utils import json_dumps


# 构建工具描述
TOOLS_DESCRIPTION = """
## Available Tools

You can use the following tools to analyze Excel data. When you need to use a tool, use this JSON format:

```json
{"tool": "tool_name", "args": {"param_name": "param_value"}}
```

### Tool List:

1. **filter_data** - Filter data by conditions (supports sorting, column selection)
   - filters (list): Multi-condition filter list, each contains column, operator, value
   - select_columns (list): Specify returned column names (optional)
   - sort_by (string): Sort column name (optional), can complete filter+sort in one step
   - ascending (bool): Sort direction, true=ascending/false=descending, default true
   - column (string): Single condition filter column name (optional)
   - operator (string): Comparison operator (==, !=, >, <, >=, <=, contains, startswith, endswith)
   - value (any type): Comparison value, supports strings, numbers, dates, etc.
   - limit (int): Return quantity limit, default 20
   - **Tip**: When filtering + sorting is needed, use this tool to complete in one step

2. **aggregate_data** - Aggregate statistics on columns (supports post-filter aggregation)
   - column (string): 【Required】Column name to aggregate
   - agg_func (string): 【Required】Aggregation function: sum, mean, count, min, max, median, std
   - filters (list): Optional filter conditions, filter first then aggregate

3. **group_and_aggregate** - Group by columns and aggregate (supports filtering)
   - group_by (string): Group column name
   - agg_column (string): Column name to aggregate
   - agg_func (string): Aggregation function (sum, mean, count, min, max)
   - filters (list): Filter conditions. **【Important】If user specifies date, region, etc., must pass here, otherwise will aggregate entire table**
   - limit (int): Return quantity limit, default 20

4. **search_data** - Search keywords in specified or all columns
   - keyword (string): Search keyword
   - columns (list): Limit search column names (optional)
   - select_columns (list): Specify returned column names
   - limit (int): Return quantity limit, default 20

5. **get_column_stats** - Get detailed column statistics (supports filtering)
   - column (string): Column name
   - filters (list): Optional filter conditions

6. **get_unique_values** - Get list of unique values in column (supports filtering)
   - column (string): Column name
   - filters (list): Optional filter conditions
   - limit (int): Return quantity limit, default 50

7. **get_data_preview** - Get data preview
   - n_rows (int): Preview row count, default 10

8. **get_current_time** - Get current system time
   - No parameters

9. **calculate** - Execute mathematical calculations (supports batch)
    - expressions (list): String format math expression list, e.g. ["(A+B)/C", "100*0.5"]

10. **generate_chart** - Generate ECharts visualization charts
    - chart_type (string): Chart type: bar, line, pie, scatter, radar, funnel, or "auto" for auto-recommendation
    - x_column (string): X-axis data column (required for bar/line charts)
    - y_column (string): Y-axis data column (numeric column)
    - group_by (string): Group column (required for pie/funnel charts)
    - agg_func (string): Aggregation function: sum, mean, count, min, max
    - title (string): Chart title
    - filters (list): Filter conditions
    - series_columns (list): Multi-series Y-axis column names (radar charts need at least 3)
    - limit (int): Data point quantity limit, default 20
    - **Use cases**: When users want to visualize data, generate charts, plot trends, show proportions

## Important Rules
- If you need to call a tool, only output one JSON object, no other text
- After tool call I will tell you the result, then you answer the user's question based on results
- If no tool is needed, answer directly in natural language
"""


SYSTEM_PROMPT_WITH_TOOLS = """You are a professional Excel data analysis assistant.

**🌍 LANGUAGE RULE (HIGHEST PRIORITY)**
TARGET RESPONSE LANGUAGE: {target_language}

You MUST respond in {target_language}, even if the spreadsheet/knowledge/tool outputs contain other languages.
You MAY quote column names or cell values in their original language, but the surrounding explanation must be in {target_language}.

## Current Excel Information
{excel_summary}

## Related Knowledge Reference
{knowledge_context}

{tools_description}

## Working Principles
1. **CRITICAL: The Excel data is ALREADY LOADED and accessible via tools. NEVER ask the user to provide data, column sums, or any information that can be obtained through tools.**
2. **When the user asks for data across multiple columns (e.g., "monthly totals", "all months"), you MUST call the tool multiple times (once per column) to gather ALL the required data before responding.**
3. Based on user questions, determine if tools are needed
4. If tools needed, **immediately call the appropriate tool** and **only output** tool call JSON, **strictly prohibit** any other text, thinking process, or explanation
5. After successful tool call, if you need more data from other columns, **immediately call the tool again** for the next column
6. After gathering ALL required data, answer user questions based on complete results
7. **In final answer, directly provide conclusions and analysis**, do not describe "I used xx tool" or "I performed xx operation" or other internal processes
8. **ALWAYS use tools to retrieve data instead of asking the user for it**
9. Maintain friendly tone and provide data analysis recommendations
10. If there is related knowledge reference, follow the rules and suggestions within
"""





def get_llm(llm_factory: Optional[LLMFactory] = None):
    """获取 LLM 实例（支持 DI 注入）"""
    if llm_factory is not None:
        return llm_factory.create_chat_model()
    # 向后兼容：使用容器
    from .core import get_container
    return get_container().get_llm_factory().create_chat_model()


def parse_tool_call(text: str) -> Dict[str, Any] | None:
    """从文本中解析工具调用 JSON（支持嵌套结构）"""
    # 尝试匹配 JSON 代码块
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 尝试提取完整的 JSON 对象（支持嵌套）
    # 找到第一个包含 "tool" 的 { 开始，然后匹配括号
    start_idx = text.find('{')
    while start_idx != -1:
        # 尝试从这个位置提取完整JSON
        depth = 0
        end_idx = start_idx
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text[start_idx:], start_idx):
            if escape_next:
                escape_next = False
                continue
            if char == '\\' and in_string:
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break
        
        if depth == 0 and end_idx > start_idx:
            candidate = text[start_idx:end_idx]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and "tool" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass
        
        # 继续找下一个 {
        start_idx = text.find('{', start_idx + 1)
    
    return None


def execute_tool(tool_name: str, tool_args: dict, tool_registry: Optional[ToolRegistry] = None) -> dict:
    """执行工具调用（支持 DI 注入）"""
    if tool_registry is not None:
        return tool_registry.execute(tool_name, tool_args)
    # 向后兼容：使用全局工具列表
    for tool in ALL_TOOLS:
        if tool.name == tool_name:
            try:
                return tool.invoke(tool_args)
            except Exception as e:
                return {"error": str(e)}
    return {"error": f"Tool not found: {tool_name}"}


async def stream_chat(message: str, history: list = None) -> AsyncGenerator[Dict[str, Any], None]:
    """执行对话
    
    Args:
        message: 当前用户消息
        history: 历史对话列表，每项为 {"role": "user"|"assistant", "content": "..."}
    """
    loader = get_loader()
    target_language = detect_target_language(message)

    if not loader.is_loaded:
        yield {
            "type": "error",
            "content": localize(
                target_language,
                en="Please upload an Excel file first.",
                zh="请先上传 Excel 文件",
            ),
        }
        return

    try:
        excel_summary = loader.get_summary(language=target_language)
        llm = get_llm()

        # 主对话
        yield {
            "type": "thinking",
            "content": localize(target_language, en="Planning...", zh="正在规划解答..."),
        }

        # 检索相关知识
        knowledge_context = format_knowledge_context([], language=target_language)
        kb = get_knowledge_base()
        if kb:
            try:
                stats = kb.get_stats()
                print(f"[知识库] 状态: {stats['total_entries']} 条知识")
                relevant_knowledge = kb.search(query=message)
                print(f"[知识库] 检索到 {len(relevant_knowledge)} 条相关知识")
                if relevant_knowledge:
                    knowledge_context = format_knowledge_context(
                        relevant_knowledge,
                        language=target_language,
                    )
                    yield {
                        "type": "thinking",
                        "content": localize(
                            target_language,
                            en=f"Found {len(relevant_knowledge)} relevant knowledge items...",
                            zh=f"找到 {len(relevant_knowledge)} 条相关知识参考...",
                        ),
                    }
            except Exception as e:
                # 知识库检索失败不影响主流程
                print(f"[知识库检索] 警告: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("[知识库] 未启用或初始化失败")
        
        system_prompt = SYSTEM_PROMPT_WITH_TOOLS.format(
            excel_summary=excel_summary,
            tools_description=TOOLS_DESCRIPTION,
            knowledge_context=knowledge_context,
            target_language=language_label(target_language),
        )
        
        # 构建对话上下文，包含历史记录
        conversation = [SystemMessage(content=system_prompt)]
        
        # 获取当前活跃表信息
        active_table_info = loader.get_active_table_info()
        current_table_name = active_table_info.filename if active_table_info else localize(
            target_language, en="Unknown table", zh="未知表"
        )
        
        # 添加历史对话（包含表名标记）
        if history:
            for h in history:
                content = h.get("content", "")
                table_name = h.get("tableName", "")
                
                # 如果历史消息有表名，且与当前表不同，添加标记
                if table_name and h.get("role") == "user":
                    tag = localize(target_language, en="For table", zh="针对表")
                    content = f"[{tag}: {table_name}] {content}"
                
                if h.get("role") == "user":
                    conversation.append(HumanMessage(content=content))
                elif h.get("role") == "assistant":
                    conversation.append(AIMessage(content=content))

        # 添加当前消息（标记当前表）
        tag = localize(target_language, en="Current table", zh="当前操作表")
        current_message = f"[{tag}: {current_table_name}] {message}"
        conversation.append(HumanMessage(content=current_message))

        # 更新 prompt - 简化指令，避免过度思考
        conversation[0].content += """
**IMPORTANT INSTRUCTIONS:**
1. If you need to use a tool to answer the question, output the tool call JSON immediately
2. Do NOT write long explanations before calling tools
3. After getting tool results, provide a clear answer in the TARGET RESPONSE LANGUAGE
4. Be direct and action-oriented
"""
        
        max_iterations = 50
        
        for _ in range(max_iterations):
            response = await llm.ainvoke(conversation)
            response_text = response.content
            
            # 解析工具调用
            tool_call = parse_tool_call(response_text)
            
            if tool_call and "tool" in tool_call:
                # 尝试提取 JSON 之前的思考文本
                thought_text = ""
                json_start = response_text.find('{')
                if json_match := re.search(r'```json', response_text):
                    thought_text = response_text[:json_match.start()].strip()
                elif json_start > 0:
                    thought_text = response_text[:json_start].strip()
                
                # 如果有思考文本且长度足够，发送更新
                if thought_text and len(thought_text) > 2:
                    yield {"type": "thinking", "content": thought_text}
                
                yield {"type": "thinking_done"}

                tool_name = tool_call["tool"]
                tool_args = tool_call.get("args", {})
                
                yield {
                    "type": "tool_call",
                    "name": tool_name,
                    "args": tool_args,
                }
                
                # 执行工具
                tool_result = execute_tool(tool_name, tool_args)
                
                yield {
                    "type": "tool_result",
                    "name": tool_name,
                    "result": tool_result,
                }
                
                # 将工具结果作为新消息继续对话
                result_message = localize(
                    target_language,
                    en=(
                        f"Tool `{tool_name}` result:\n```json\n"
                        f"{json_dumps(tool_result, ensure_ascii=False, indent=2)}\n```\n\n"
                        f"Answer the user's question using this result. Respond in {language_label(target_language)}."
                    ),
                    zh=(
                        f"工具 `{tool_name}` 执行结果：\n```json\n"
                        f"{json_dumps(tool_result, ensure_ascii=False, indent=2)}\n```\n\n"
                        "请根据这个结果回答用户的问题。"
                    ),
                )

                # Only persist the clean JSON tool call (avoid leaking non-JSON chatter).
                conversation.append(AIMessage(content=json_dumps(tool_call, ensure_ascii=False)))
                conversation.append(HumanMessage(content=result_message))
                
            else:
                # 没有工具调用，直接输出响应
                final_text = response_text
                print(f"[Language Check] Target: {target_language}, Response length: {len(final_text)}")
                mismatch = is_language_mismatch(target_language, final_text)
                print(f"[Language Check] Is mismatch: {mismatch}")

                if mismatch:
                    print(f"[Language Rewrite] Rewriting to {language_label(target_language)}")
                    rewritten = await llm.ainvoke(
                        [
                            SystemMessage(content=rewrite_system_prompt(target_language)),
                            HumanMessage(content=final_text),
                        ]
                    )
                    if isinstance(rewritten, AIMessage) and rewritten.content:
                        final_text = rewritten.content
                        print(f"[Language Rewrite] Success, new length: {len(final_text)}")

                yield {"type": "token", "content": final_text}
                yield {"type": "done", "content": final_text}
                return

        yield {
            "type": "error",
            "content": localize(target_language, en="Reached max iterations.", zh="达到最大迭代次数"),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield {"type": "thinking_done"}
        yield {
            "type": "error",
            "content": localize(
                target_language,
                en=f"Error: {str(e)}",
                zh=f"处理出错: {str(e)}",
            ),
        }
