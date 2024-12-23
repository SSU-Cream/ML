# -*- coding: utf-8 -*- 
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

np.random.seed(55)

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

FACE_PREDICTOR_PATH = os.path.join(CURRENT_PATH,'..','common','predictors','shape_predictor_68_face_landmarks.dat')

PREDICT_GREEDY      = True
PREDICT_BEAM_WIDTH  = 200
PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'..','common','dictionaries','grid.txt')

def predict(weight_path, video_path, absolute_max_string_len=32, output_size=11174):
    print "\nLoading data from disk..."
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
    if os.path.isfile(video_path):
        video.from_video(video_path)
    else:
        video.from_frames(video_path)
    print "Data loaded.\n"

    if K.image_data_format() == 'channels_first':
        # print "a"
        img_c, frames_n, img_w, img_h = video.data.shape
    else:
        # print "b"
        frames_n, img_w, img_h, img_c = video.data.shape

    # print "c"

    lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                    absolute_max_string_len=absolute_max_string_len, output_size=output_size)

    lipnet.summary()
    # print "d"

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # print "e"
    lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
    # print "f"

    lipnet.model.load_weights(weight_path)

    # print "g"
    spell = Spell(path=PREDICT_DICTIONARY)
    # print "h"
    decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                      postprocessors=[labels_to_text, spell.sentence])
    # print "i"
    X_data = np.array([video.data]).astype(np.float32) / 255
    input_length = np.array([len(video.data)])
    y_pred = lipnet.predict(X_data)

    #print X_data
    #print "========= x_pred ============"
    #print y_pred
    #print "========= y_pred ============"
    #print input_length
    #print "========= input_length ============"
    result = decoder.decode(y_pred, input_length)[0]
    #print result
    print "========= result ============"

    if(result == u'\uae30\uc5ed'):
        a
        #result = "기역"

    elif(result == u'\ub2c8\uc740'):
        a
        #result = "니은"

    elif(result == u'\ub514\uadff'):
        a
        #result = "디귿"

    elif(result == u'\ub9ac\uc744'):
        a
        #result = "리을"

    elif(result == u'\ubbf8\uc74c'):
        a
        #result = "미음"

    elif(result == u'\ube44\uc74d'):
        a
        #result = "비읍"

    elif(result == u'\uc2dc\uc637'):
        a
        #result = "시옷"

    elif(result == u'\uc774\uc751'):
        a
        #result = "이응"

    elif(result == u'\uc9c0\uc752'):
        a
        #result = "지읒"

    elif(result == u'\uce58\uc753'):
        a
        #result = "치읓"

    elif(result == u'\ud0a4\uc754'):
        a
        #result = "키읔"

    elif(result == u'\ud2f0\uc755'):
        a
        #result = "티읕"

    elif(result == u'\ud53c\uc756'):
        a
        #result = "피읖"

    elif(result == u'\ud788\uc757'):
        a
        #result = "히읗"
    else:
        #result = "기역"
        result = u'\uae30\uc5ed'

    return (video, result)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        video, result = predict('./weights952.h5', './post/video.mp4')
    elif len(sys.argv) == 3:
        video, result = predict(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        video, result = predict(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5:
        video, result = predict(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        video, result = None, ""

    if video is not None:
        show_video_subtitle(video.face, result)

    stripe = "-" * len(result)
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
    print "             --{}- ".format(stripe)
    print "[ DECODED ] |> {} |".format(result.encode('utf-8'))
    print "             --{}- ".format(stripe)

    fp = open('./result.txt', 'wt')
    fp.write(result.encode('utf-8'))
    fp.close()
