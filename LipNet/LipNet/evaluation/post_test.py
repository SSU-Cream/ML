# -*- encoding: utf-8 -*-
from flask import Flask, request
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

global result
result = '예측 전'

@app.route('/')
def hello():
    return (result)


@app.route('/predict',methods=['POST'])
def post():
    post_file = request.files['file']
    filename = secure_filename(post_file.filename)
    print("post filename >>>>>>> " + filename)

    CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
    #print("path >> " + CURRENT_PATH)
    post_file.save(os.path.join(CURRENT_PATH,'post',filename))

    global result
    result = 'OK'
    return result

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3389)
