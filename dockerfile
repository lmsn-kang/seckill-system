FROM python:3.9-slim
RUN pip install --no-cache-dir pandas numpy matplotlib scipy pymysql
WORKDIR /app