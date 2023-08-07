FROM public.ecr.aws/lambda/python:3.9.1

COPY ./ /app

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY



ENV GIT_PYTHON_REFRESH=quiet
WORKDIR /app/deploy

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \ 
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

RUN pip install "dvc[s3]"
RUN pip install -r requirements.txt
RUN dvc init --no-scm
RUN dvc remote add -d model-store s3://mlopshungvo/
RUN dvc pull ../deploy/onnx_pretrain_model/cola_epoch_0.onnx.dvc


EXPOSE 8000

ENV MODULE_NAME="server"


RUN python lambda_handler.py
CMD [ "lambda_handler.lambda_handler"]
