#FROM lukewiwa/aws-lambda-python-sqlite:3.9
FROM amazon/aws-lambda-python:3.9.2023.08.02.09
COPY ./ /app

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

WORKDIR /app/deploy/

ENV GIT_PYTHON_REFRESH=quiet

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \ 
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

 RUN wget "https://www.sqlite.org/2018/sqlite-autoconf-3260000.tar.gz" && tar xzf sqlite-autoconf-3260000.tar.gz && cd sqlite-autoconf-3260000 && ./configure --disable-static --enable-fts5 --enable-json1 CFLAGS="-g -O2 -DSQLITE_ENABLE_FTS3=1 -DSQLITE_ENABLE_FTS4=1 -DSQLITE_ENABLE_RTREE=1 -DSQLITE_ENABLE_JSON1" && make && make install 
RUN pip install "dvc[s3]"

RUN pip install -r requirements.txt
RUN dvc init --no-scm
RUN dvc remote add -d model-store s3://mlopshungvo/
RUN dvc pull ../deploy/onnx_pretrain_model/cola_epoch_0.onnx.dvc



ENV MODULE_NAME="server"

COPY deploy/lambda_function.py ${LAMBDA_TASK_ROOT}/
EXPOSE 8000


CMD ["lambda_function.lambda_handler"]
#CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "reload"]
