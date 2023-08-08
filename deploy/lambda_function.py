import sys
import json
import boto3

sys.path.append("../exps")
from extract_config import extract_config
from onnx_inference import ColaONNXPredictor

onnx_model_path = r"onnx_pretrain_model/cola_epoch_0.onnx"
predictor = ColaONNXPredictor(onnx_model_path)


regions = ["us-east-1"]
autoscaling = boto3.client('autoscaling')

def lambda_handler(event, context):
    """
    Lambda function handler for predicting linguistic acceptability of the given sentence
    """
    response = autoscaling.describe_auto_scaling_groups(MaxRecords=100)
  #print response
  #print(response['AutoScalingGroups'][0]['AutoScalingGroupName'])
    autoscaling_group_to_suspend = []
    for doc in response['AutoScalingGroups']:
        response_parsed = doc['AutoScalingGroupName']
        autoscaling_group_to_suspend.append(response_parsed)

  #print autoscaling_group_to_suspend
    import re
    regex = re.compile(r'es-data-asg|consul|influxdb|vault|es-master|compress')
    filtered = filter(lambda i: not regex.search(i), autoscaling_group_to_suspend)
    filtered = [i for i in autoscaling_group_to_suspend if not regex.search(i)]
    

    if len(filtered) > 0:
        for x in filtered:
            autoscaling.suspend_processes(AutoScalingGroupName=x)
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