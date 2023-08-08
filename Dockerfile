FROM python:3.9.1-slim as base
#FROM lukewiwa/aws-lambda-python-sqlite:3.9
#FROM amazon/aws-lambda-python:3.9.2023.08.02.09
COPY ./ /app

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

WORKDIR /app/deploy/

ENV GIT_PYTHON_REFRESH=quiet

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \ 
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY


RUN pip install "dvc[s3]"
RUN pip install -r requirements.txt 
RUN dvc init --no-scm
RUN dvc remote add -d model-store s3://mlopshungvo/
RUN dvc pull ../deploy/onnx_pretrain_model/cola_epoch_0.onnx.dvc

ENV MODULE_NAME="server"

EXPOSE 8000


#CMD ["lambda_function.lambda_handler"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "reload"]
