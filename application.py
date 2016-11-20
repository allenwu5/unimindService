__author__ = 'len'

from flask import Flask

from sys import path
from os import getcwd
path.append(getcwd() + "/keras") #Yes, i'm on windows
print path
import my_cifar as cf

import boto3
import cStringIO

# EB looks for an 'application' callable by default.
application = Flask(__name__)

@application.route('/cifar')
def cifar():
    cf.path = 'keras/'
    img_names = ['standing-cat.jpg', 'dog-face.jpg', 'bird.jpeg', 'car.jpeg', 'truck.jpeg', 'ape.jpg', 'duck.jpg', 'mustbebird.jpeg',
             'cnBird.jpeg', 'cnBird2.jpeg', '3birds.jpeg'
             ]
    for i in xrange(len(img_names)):
        img_names[i] = cf.path + '../cifar_images/' + img_names[i]
    imgs = cf.load_and_scale_imgs(img_names)
    return cf.classify(img_names, imgs)


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

    img_names = []

    html = ""
    for obj in bucket.objects.all():
        key = obj.key

        # only debugging specific case now
        if key == 'public/w32/1472541858.06554/84.jpeg':
            body = obj.get()['Body'].read()
            img = cStringIO.StringIO(body)
            img_names.append(img)


    cf.path = 'keras/'

    # for i in xrange(len(img_names)):
    #     img_names[i] = cf.path + '../cifar_images/' + img_names[i]
    imgs = cf.load_and_scale_imgs(img_names)
    img_names = [key]
    result = cf.classify(img_names, imgs)

    # upload result to
    return result

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    # application.debug = True
    # application.run()
    print(root())