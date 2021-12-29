# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:25:10 2020

@author: jfili
"""

from flask import Flask, request, render_template, flash, redirect, url_for
from flask_restful import Resource, Api,reqparse
import requests
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
import base64
import json
from datetime import datetime

app = Flask(__name__)
api = Api(app)

UPLOAD_FOLDER = 'images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')
def index():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class Detection(Resource):
    
    
    def post(self):
        
        y = json.loads(request.data)
        b = bytes(y['file'], 'utf-8')
        today = datetime.now()
        # if user does not select file, browser also
        # submit an empty part without filename
        dt_string = today.strftime("%d%m%Y%H%M%S")
        filename = dt_string+'.jpg'
        filename2 = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filename2, 'wb') as file_to_save:
            decoded_image_data = base64.decodebytes(b)
            file_to_save.write(decoded_image_data)
        
        response=analyze(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return response
    

api.add_resource(Detection, '/Detection')
def analyze(image):
    args={"face" : "model", "model" : "model/mask_detector.model",
    "image" : image, "confidence" : 0.5}

    # load our serialized face detector model from disk
    net = cv2.dnn.readNet("model/deploy.prototxt", "model/res10_300x300_ssd_iter_140000.caffemodel")
        
    # load the face mask detector model from disk
    model = load_model(args["model"])
        
    # load the input image from disk, clone it, and grab the image spatial
    # dimensions
    image = cv2.imread(args["image"])
    orig = image.copy()
    (h, w) = image.shape[:2]
    
    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
    	(104.0, 177.0, 123.0))
    
    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
        a=0
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]
    
    	# filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
      		# compute the (x, y)-coordinates of the bounding box for
      		# the object
              box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
              (startX, startY, endX, endY) = box.astype("int")
      
      		# ensure the bounding boxes fall within the dimensions of
      		# the frame
              (startX, startY) = (max(0, startX), max(0, startY))
              (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
      
      		# extract the face ROI, convert it from BGR to RGB channel
      		# ordering, resize it to 224x224, and preprocess it
              face = image[startY:endY, startX:endX]
              face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
              face = cv2.resize(face, (224, 224))
              face = img_to_array(face)
              face = preprocess_input(face)
              face = np.expand_dims(face, axis=0)
      
      		# pass the face through the model to determine if the face
      		# has a mask or not
              (mask, withoutMask) = model.predict(face)[0]
      
      		# determine the class label and color we'll use to draw
      		# the bounding box and text
              label = "Mask" if mask > withoutMask else "NoMask"
      		
      		# include the probability in the label
              objectLabel={"object{}".format(a):{ "startX":int(startX), "startY": int(startY), "endX":int(endX), "endY":int(endY), label: max(mask, withoutMask) * 100 }}
              orig={}
              orig.update(objectLabel)
              a=a+1
    return orig

if __name__ == '__main__':
    app.run(debug=True)
    