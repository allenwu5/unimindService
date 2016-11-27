__author__ = 'len'

from flask import Flask

from sys import path
from os import getcwd
path.append(getcwd() + "/keras") #Yes, i'm on windows
print(path)
import my_cifar as cf

from image.cifar_image import CifarImage

import boto3
from io import BytesIO

# EB looks for an 'application' callable by default.
application = Flask(__name__)

@application.route('/s3')
def s3():
    # Let's use Amazon S3
    s3 = boto3.resource('s3')

    bucket = s3.Bucket('unimind-userfiles-mobilehub-1656990244')

    html = ""
    for obj in bucket.objects.all():
        key = obj.key

        # only debugging specific case now
        if key == 'public/w32/1472541858.06554/87.jpeg':
            body = obj.get()['Body'].read()


        html += key + "<br>"
    return html


@application.route('/')
def root():
    # Let's use Amazon S3
    s3 = boto3.resource('s3')

    bucket = s3.Bucket('unimind-userfiles-mobilehub-1656990244')

    cifar_imgs = []

    for obj in bucket.objects.all():
        key = obj.key

        # only debugging specific case now
        if key.startswith('public/w32/1472541858.06554/'):
            body = obj.get()['Body'].read()
            #
            ci = CifarImage()
            ci.name = key
            ci.body = BytesIO(body)
            cifar_imgs.append(ci)

    cf.path = 'keras/'
    pre_processed_imgs = cf.pre_process(cifar_imgs)
    result = cf.classify(cifar_imgs, pre_processed_imgs)


    # upload result to ...
    return result

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    # application.debug = True
    # application.run()
    print(root())