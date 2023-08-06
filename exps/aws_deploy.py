import boto3

s3 = boto3.resource('s3')
bucket = s3.Bucket('ita-test-01')
for obj in bucket.objects.all():
    print(obj.key)