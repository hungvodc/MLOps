FROM python:3.9.1-slim as base

COPY ./ /app

WORKDIR /app/deploy

RUN pip install -r requirements.txt

EXPOSE 8000

ENV MODULE_NAME="server"


CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]