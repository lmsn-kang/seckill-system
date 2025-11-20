import json
import re
from typing import TypedDict, Annotated, List
import operator

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI

from config import OPENAI_MODEL
from tools.database_tool import db, query_database
from agents.workers import sql_chain, analyzer_chain, report_chain
from agents.mcp_manager import create_tool_dynamically


class AgentState(TypedDict):
    user_query: str
    messages: Annotated[List[AIMessage | HumanMessage], operator.add]
    sql_query: str
    data_json: str     
    analysis_log: str   
    final_report: str
    tool_output: str    


planner_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

def planning_node(state: AgentState):
   
    last_msg = state['messages'][-1].content if state['messages'] else "无"
    
    prompt = f"""
    你是 AgentOrchestra 的总指挥。
    目标：{state['user_query']}
    
    当前状态/最新结果：
    {last_msg[:2000]}... (略)
    
    请决定下一步行动，返回 JSON：
    {{
      "next_step": "sql_generator" (查数据) | "mcp_analysis" (Python处理数据) | "deep_analyzer" (解读数据) | "report_generator" (生成报告并结束),
      "reason": "为什么做这一步",
      "mcp_task": "如果选 mcp_analysis，描述需要计算什么（例如：计算加权平均库存周转率）"
    }}
    
    规则：
    1. 如果还没有数据，先去 sql_generator。
    2. 如果有了 SQL 结果，但需要复杂计算（如相关性系数、复杂清洗），去 mcp_analysis。
    3. 如果数据已经足够，去 deep_analyzer 进行业务解读。
    4. 解读完毕去 report_generator。
    """
    
    try:
        response = planner_llm.invoke(prompt).content
        plan = json.loads(response.replace("```json", "").replace("```", ""))
    except:
        
        plan = {"next_step": "report_generator", "reason": "解析失败，强制结束"}
        
    return {
        "messages": [AIMessage(content=f"指挥官决策: {plan['reason']}")],
        "tool_output": json.dumps(plan) # 借用字段传递 plan
    }

def router(state: AgentState):
    plan = json.loads(state["tool_output"])
    return plan["next_step"]



def node_sql(state: AgentState):
    query = sql_chain.invoke({"user_query": state["user_query"], "table_info": db.get_table_info()})
    return {"sql_query": query, "messages": [AIMessage(content=f"生成 SQL: {query}")]}

def node_executor(state: AgentState):
    
    result = query_database.invoke({"sql": state["sql_query"]})
    
   
    data_json = ""
    if "<<DATA_JSON>>" in result:
        parts = result.split("<<DATA_JSON>>")
        text_res = parts[0]
        data_json = parts[1]
    else:
        text_res = result
    
    return {
        "messages": [AIMessage(content=f"数据库结果: {text_res}")],
        "data_json": data_json
    }

def node_mcp(state: AgentState):
    """核心：动态创建 Docker 工具并执行"""
    plan = json.loads(state["tool_output"])
    task_need = plan.get("mcp_task", "通用数据处理")
    
    if not state.get("data_json"):
        return {"messages": [AIMessage(content="错误：没有数据可供 MCP 处理，请先执行 SQL。")]}

    
    custom_tool = create_tool_dynamically(task_need, state["data_json"])
    
   
    print(f"--- [MCP] 正在构建 Docker 工具: {task_need} ---")
    tool_result = custom_tool.invoke({"input_json": state["data_json"]})
    
    return {
        "messages": [AIMessage(content=f"MCP (Docker) 分析结果:\n{tool_result}")],
        "analysis_log": state.get("analysis_log", "") + f"\n\nPython计算结果：\n{tool_result}"
    }

def node_analyzer(state: AgentState):
    
    context = state['messages'][-1].content
    analysis = analyzer_chain.invoke({"user_query": state["user_query"], "data_result": context})
    return {
        "analysis_log": state.get("analysis_log", "") + f"\n\n分析师洞察：\n{analysis}",
        "messages": [AIMessage(content=analysis)]
    }

def node_report(state: AgentState):
    report = report_chain.invoke({
        "user_query": state["user_query"], 
        "analysis_context": state["analysis_log"]
    })
    return {"final_report": report}


workflow = StateGraph(AgentState)
memory = SqliteSaver.from_conn_string(":memory:")

workflow.add_node("planner", planning_node)
workflow.add_node("sql_generator", node_sql)
workflow.add_node("executor", node_executor)
workflow.add_node("mcp_analysis", node_mcp)
workflow.add_node("deep_analyzer", node_analyzer)
workflow.add_node("report_generator", node_report)


workflow.set_entry_point("planner")

workflow.add_conditional_edges("planner", router, {
    "sql_generator": "sql_generator",
    "mcp_analysis": "mcp_analysis",
    "deep_analyzer": "deep_analyzer",
    "report_generator": "report_generator"
})

workflow.add_edge("sql_generator", "executor")
workflow.add_edge("executor", "planner")      
workflow.add_edge("mcp_analysis", "planner")  
workflow.add_edge("deep_analyzer", "planner") 
workflow.add_edge("report_generator", END)

app = workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    import uuid
    thread_id = str(uuid.uuid4())
    
    print("=== AgentOrchestra (Docker Safe Mode) 启动 ===")
    user_input = input("请输入需求 : ")
    
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {
        "user_query": user_input, 
        "messages": [],
        "analysis_log": ""
    }
    
    for event in app.stream(inputs, config):
        for node, values in event.items():
            print(f"\n--- Node: {node} ---")
            if "messages" in values:
                print(values["messages"][-1].content[:200] + "...")
            if "final_report" in values:
                print("\n" + "="*30 + " 最终报告 " + "="*30)
                print(values["final_report"])