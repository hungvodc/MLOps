import sys
import json

sys.path.append("../exps")
from extract_config import extract_config
from onnx_inference import ColaONNXPredictor

onnx_model_path = r"onnx_pretrain_model/cola_epoch_0.onnx"
predictor = ColaONNXPredictor(onnx_model_path)

def lambda_handler(event, context):
    """
    Lambda function handler for predicting linguistic acceptability of the given sentence
    """

    if "resource" in event.keys():
        body = event["body"]
        body = json.loads(body)
        print(f"Got the input: {body['sentence']}")
        response = predictor.predict(body["sentence"])
        return {
            "statusCode": 200,
            "headers": {},
            "body": json.dumps(response[0])
        }
    else:
        return predictor.predict(event["sentence"])
    
if __name__ == "__main__":
	test = {"sentence": "this is a sample sentence"}
	print(lambda_handler(test, None))