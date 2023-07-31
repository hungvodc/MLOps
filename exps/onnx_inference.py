import numpy as np
import onnxruntime as ort

from data import DataModule
from extract_config import extract_config

config = extract_config()

class ColaONNXPredictor:
    def __init__(self, model_path):
        self.ort_session = ort.InferenceSession(model_path)
        self.processor = DataModule()
        self.labels = ["unacceptable", "acceptable"]

    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)
        
        # expand dim for batch size
        ort_inputs = {
            "input_ids": (np.expand_dims(processed["input_ids"], axis=0)).astype('int64'),
            "attention_mask": np.expand_dims(processed["attention_mask"], axis=0).astype('int64'),
        }
        
        ort_outs = self.ort_session.run(None, ort_inputs)
        predictions = []
        for ele in ort_outs:
            if ele > config["threshold"]:
                predictions.append(self.labels[1])
            else:
                predictions.append(self.labels[0])
        
        for score in ort_outs:
            predictions.append({"score": score[0][0]})
        return predictions


if __name__ == "__main__":
    sentence = config["example_sentence"]
    onnx_model_path = config["onnx_converter_model"]
    predictor = ColaONNXPredictor(onnx_model_path)
    print(predictor.predict(sentence))