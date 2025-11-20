from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import OPENAI_MODEL
from tools.database_tool import db

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)


sql_prompt = PromptTemplate.from_template("""
你是一个 MySQL 数据库专家。
表结构如下：
{table_info}

用户需求：{user_query}

请生成一条可执行的 MySQL SQL 语句。
注意：
1. 使用 MySQL 语法（例如使用 `LIMIT` 而不是 `TOP`）。
2. 日期处理使用 MySQL 函数（如 `DATE_FORMAT`, `DATEDIFF`）。
3. 只输出 SQL，不要 markdown，不要解释。
""")
sql_chain = sql_prompt | llm | StrOutputParser()

# --- Deep Analyzer ---
analyzer_prompt = PromptTemplate.from_template("""
你是一名数据分析师。
用户需求：{user_query}
数据库/工具返回结果：
{data_result}

请分析上述数据，提取关键洞察。
如果数据中包含 "<<DATA_JSON>>" 标记，请忽略该标记后的原始数据，仅根据前面的摘要或工具的分析结果进行解读。
重点关注异常值、趋势和关联性。
""")
analyzer_chain = analyzer_prompt | llm | StrOutputParser()

# --- Report Generator ---
report_prompt = PromptTemplate.from_template("""
作为电商运营总监，请根据以下信息生成最终报告。

用户原始需求：{user_query}
分析过程与结论：{analysis_context}

生成一份 Markdown 格式的报告，包含：
1. 核心结论
2. 数据支撑 (引用分析结果)
3. 具体的业务建议
""")
report_chain = report_prompt | llm | StrOutputParser()