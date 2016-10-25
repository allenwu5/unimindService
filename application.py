__author__ = 'len'

from flask import Flask
import boto3

# EB looks for an 'application' callable by default.
application = Flask(__name__)

@application.route('/')
def test_s3():
    # Let's use Amazon S3
    s3 = boto3.resource('s3')
    # client = boto3.client(
    #     's3',
    #     # Hard coded strings as credentials, not recommended.
    #     # us-west-2
    #     aws_access_key_id='AKIAJWDOXVJRFPKTBQNQ',
    #     aws_secret_access_key='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'
    # )

    bucket = s3.Bucket('unimind-userfiles-mobilehub-1656990244')

    result = ""
    for obj in bucket.objects.all():
        key = obj.key
        # body = obj.get()['Body'].read()

        result += key + "<br>"
    return result

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()