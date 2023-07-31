import sys
from fastapi import FastAPI

sys.path.append('../exps')

from onnx_inference import ColaONNXPredictor
from extract_config import extract_config

app = FastAPI()

onnx_model_path = r'onnx_pretrain_model/cola_epoch_0.onnx'
predictor = ColaONNXPredictor(onnx_model_path)

@app.get("/")
async def home():
    return "<h2>This is an NLP project<\h2>"

@app.get("/predict")
async def get_prediction(text: str):
    result = predictor.predict(text)
    return result[0]