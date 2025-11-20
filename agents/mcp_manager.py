import docker
import textwrap
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import StructuredTool
from config import OPENAI_MODEL

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
docker_client = docker.from_env()

code_gen_prompt = PromptTemplate.from_template("""
你是一个 Python 数据处理专家。
任务需求：{task_need}
数据样例（第一行）：{data_sample}

请编写一段 Python 代码逻辑，对变量 `df` (pandas DataFrame) 进行处理。
要求：
1. 假设 `df` 已经存在，直接对 `df` 进行操作。
2. 最后必须将结果打印出来（print）。
3. 不要包含 `import pandas` 或读取数据的代码，只写核心处理逻辑。
4. 代码不要用 markdown 包裹，直接输出代码文本。
""")

def create_tool_dynamically(task_need: str, data_context_json: str, suggested_name: str = "custom_analysis_tool"):
    try:
        data_list = json.loads(data_context_json)
        sample = data_list[0] if len(data_list) > 0 else "No data"
    except:
        sample = "Structure unknown"

    chain = code_gen_prompt | llm | StrOutputParser()
    core_logic_code = chain.invoke({
        "task_need": task_need,
        "data_sample": str(sample)
    })

    def docker_sandbox_runner(input_json: str = data_context_json) -> str:
        full_script = textwrap.dedent(f"""
        import pandas as pd
        import json
        import sys

        try:
            raw_data = {json.dumps(input_json)} 
            if isinstance(raw_data, str):
                raw_data = json.loads(raw_data)
            
            df = pd.DataFrame(raw_data)

            print("--- Analysis Start ---")
            {textwrap.indent(core_logic_code, '            ')}
            print("--- Analysis End ---")

        except Exception as e:
            print(f"Execution Error: {{e}}")
        """)

        try:
            container = docker_client.containers.run(
                image="agent-sandbox:latest",
                command=["python", "-c", full_script],
                remove=True,
                network_disabled=True,
                mem_limit="512m"
            )
            return container.decode("utf-8")
        except Exception as e:
            return f"Docker Environment Error: {str(e)}"

    return StructuredTool.from_function(
        func=docker_sandbox_runner,
        name=suggested_name,
        description=f"Dynamic Sandbox Tool for: {task_need}"
    )