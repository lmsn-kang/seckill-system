from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.tools import tool
from config import DATABASE_URL
import pandas as pd
from sqlalchemy import create_engine

db = SQLDatabase.from_uri(DATABASE_URL)
engine = create_engine(DATABASE_URL)

@tool
def query_database(sql: str) -> str:
    try:
        clean_sql = sql.replace("```sql", "").replace("```", "").strip()
        df = pd.read_sql(clean_sql, engine)
        
        if df.empty:
            return "Query executed successfully but returned no data."
        
        summary = df.head().to_markdown()
        json_data = df.to_json(orient='records')
        
        return f"Summary:\n{summary}\n\n<<DATA_JSON>>{json_data}<<DATA_JSON>>"
        
    except Exception as e:
        return f"MySQL Execution Error: {str(e)}"