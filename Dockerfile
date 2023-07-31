FROM python:3.9.1-slim as base

COPY ./ /app

ENV GIT_PYTHON_REFRESH=quiet
WORKDIR /app/deploy

RUN pip install "dvc[gdrive]"
RUN pip install -r requirements.txt
RUN dvc init --no-scm
RUN dvc remote add -d storage gdrive://19JK5AFbqOBlrFVwDHjTrf9uvQFtS0954
RUN dvc remote modify storage gdrive_use_service_account true
RUN dvc remote modify storage gdrive_service_account_json_file_path exps/config/creds.json


EXPOSE 8000

ENV MODULE_NAME="server"


CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
