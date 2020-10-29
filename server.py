# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:25:10 2020

@author: jfili
"""

from flask import Flask, request, render_template
from flask_restful import Resource, Api,reqparse
import requests
app = Flask(__name__)
api = Api(app)

@app.route('/')
def index():
    return render_template('index.html')

parser = reqparse.RequestParser()
parser.add_argument('photo')
parser.add_argument('name')


class Detection(Resource):
    def post(self):
        args = parser.parse_args()
        currency=args['photo']
        transaction=args['name']
        try:
            #here will be detection
            return {"rate": 0}
        except:
            return {"rate": "error"}
        

api.add_resource(Detection, '/Detection')

if __name__ == '__main__':
    app.run(debug=True)
    