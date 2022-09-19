#!/usr/bin/env python
# -*- coding: utf-8 -*-
from server import Flask
import numpy as np
import datetime
import os
import sys
import tensorflow as tf
import cv2
app = Flask(__name__)
@app.route("/koreanlip")
def predict():
    model = tf.keras.model.load_model('/home/drop_pgs/SSU/SSUCream.h5')
    test_video = cv2.VideoCapture('/home/drop_pgs/SSU/1.mp4')
    width = int(cap.get(360))
    height = int(cap.get(288))
    fps = 25
    fcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('output.avi', fcc, fps, (width, height))
    while (cap.isOpened()) :
        ret, frame = cap.read()
        if ret :
            frame = cv2.flip(frame, 0)
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q') : break
        else :
            print("Fail to read frame!")
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    predict = model.predict(test_video)
    return jsonify(str(predict))
# if __name__ == '__main__':
#     app.run()
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run(host="0.0.0.0")
