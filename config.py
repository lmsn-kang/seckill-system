import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL =os.getenv(OPENAI_MODEL)
DATABASE_URL = os.getenv("DATABASE_URL")