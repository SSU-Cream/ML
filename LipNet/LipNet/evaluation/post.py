# -*- encoding: utf-8 -*-
from flask import Flask, request
from werkzeug.utils import secure_filename
from lipnet.lipreading.videos import Video
from lipnet.lipreading.visualization import show_video_subtitle
from lipnet.core.decoders import Decoder
from lipnet.lipreading.helpers import labels_to_text
from lipnet.utils.spell import Spell
from lipnet.model2 import LipNet
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import sys
import os

app = Flask(__name__)

np.random.seed(55)
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
FACE_PREDICTOR_PATH = os.path.join(CURRENT_PATH,'..','common','predictors','shape_predictor_68_face_landmarks.dat')
PREDICT_GREEDY      = True
PREDICT_BEAM_WIDTH  = 200
PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'..','common','dictionaries','grid.txt')


global result
result = '예측 전'

@app.route('/')
def hello():
    return (result)

@app.route('/predict',methods=['POST'])
def post():
    post_file = request.files['file']
    filename = secure_filename(post_file.filename)
    print("Request 받음")
    print("안드로이드에서 전송된 파일 : " + filename)
    post_file.save(os.path.join(CURRENT_PATH,'post',filename))
#    post_file.save(os.path.join(CURRENT_PATH,'post','video.mp4'))

    # 가중치 파일 경로
    weight_path = os.path.join(CURRENT_PATH,'weights952.h5')
    # Post 방식으로 받아온 영상 경로 지정해야함
    video_path = os.path.join(CURRENT_PATH,'post',filename)

    global result
    result = predict(weight_path, video_path)
    #execfile("predict2.py")
    K.tensorflow_backend.clear_session()

#    fp = open('./result.txt', 'r')
#    result = fp.read()

    return (result)

@app.route('/predict2')
def predict(weight_path, video_path, absolute_max_string_len=32, output_size=11174):
    absolute_max_string_len=32
    output_size=11174

    print "\nLoading data from disk..."
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
    if os.path.isfile(video_path):
        video.from_video(video_path)
    else:
        video.from_frames(video_path)
    print "Data loaded.\n"

    if K.image_data_format() == 'channels_first':
        img_c, frames_n, img_w, img_h = video.data.shape
    else:
        frames_n, img_w, img_h, img_c = video.data.shape

    lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                    absolute_max_string_len=absolute_max_string_len, output_size=output_size)
    lipnet.summary()

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)

    lipnet.model.load_weights(weight_path)

    spell = Spell(path=PREDICT_DICTIONARY)

    decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                      postprocessors=[labels_to_text, spell.sentence])

    X_data = np.array([video.data]).astype(np.float32) / 255
    input_length = np.array([len(video.data)])
    y_pred = lipnet.predict(X_data)

    global result
    result = decoder.decode(y_pred, input_length)[0]
    print "========= result ============"

    # 안드로이드에 result 전송하는 코드 작성
    if(result == u'\uae30\uc5ed'):
        result = "기역"

    elif(result == u'\ub2c8\uc740'):
        result = "니은"

    elif(result == u'\ub514\uadff'):
        result = "디귿"

    elif(result == u'\ub9ac\uc744'):
        result = "리을"

    elif(result == u'\ubbf8\uc74c'):
        result = "미음"

    elif(result == u'\ube44\uc74d'):
        result = "비읍"

    elif(result == u'\uc2dc\uc637'):
        result = "시옷"

    elif(result == u'\uc774\uc751'):
        result = "이응"

    elif(result == u'\uc9c0\uc752'):
        result = "지읒"

    elif(result == u'\uce58\uc753'):
        result = "치읓"

    elif(result == u'\ud0a4\uc754'):
        result = "키읔"

    elif(result == u'\ud2f0\uc755'):
        result = "티읕"

    elif(result == u'\ud53c\uc756'):
        result = "피읖"

    elif(result == u'\ud788\uc757'):
        result = "히읗"

    elif(result == u'\ube44'):
        result = "비읍"
    else:
        result = "기역"

    print(result)
    print ""
    print " __                   __  __          __      "
    print "/\\ \\       __        /\\ \\/\\ \\        /\\ \\__   "
    print "\\ \\ \\     /\\_\\  _____\\ \\ `\\\\ \\     __\\ \\ ,_\\  "
    print " \\ \\ \\  __\\/\\ \\/\\ '__`\\ \\ , ` \\  /'__`\\ \\ \\/  "
    print "  \\ \\ \\L\\ \\\\ \\ \\ \\ \\L\\ \\ \\ \\`\\ \\/\\  __/\\ \\ \\_ "
    print "   \\ \\____/ \\ \\_\\ \\ ,__/\\ \\_\\ \\_\\ \\____\\\\ \\__\\"
    print "    \\/___/   \\/_/\\ \\ \\/  \\/_/\\/_/\\/____/ \\/__/"
    print "                  \\ \\_\\                       "
    print "                   \\/_/                       "
    print ""


    print(result)
    return (result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3389)
