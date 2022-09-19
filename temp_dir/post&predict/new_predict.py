# -*- coding: utf-8 -*- 
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import sys
import os

from scipy import ndimage
from scipy.misc import imresize
import skvideo.io
import dlib

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.font_manager as fm

import re
import string
import io
from collections import Counter
import codecs

from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.pooling import MaxPooling3D
from keras.layers.core import Dense, Activation, SpatialDropout3D, Flatten
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from keras.layers import Input
from keras.models import Model

from keras.layers.core import Lambda
from lipnet.core.loss import ctc_lambda_func


np.random.seed(55)

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

FACE_PREDICTOR_PATH = os.path.join(CURRENT_PATH,'shape_predictor_68_face_landmarks.dat')

PREDICT_GREEDY      = True
PREDICT_BEAM_WIDTH  = 200
PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'grid.txt')


class Video(object):
    def __init__(self, vtype='mouth', face_predictor_path=None):
        if vtype == 'face' and face_predictor_path is None:
            raise AttributeError('Face video need to be accompanied with face predictor')
        self.face_predictor_path = face_predictor_path
        self.vtype = vtype

    def from_frames(self, path):
        #print(">>" + path)
        path = path.replace("kykymouse", "sun")
        #path = "/home/sun/LipNet/LipNet/training/overlapped_speakers/s1/datasets/train/s1/csbfo"
        print("\npath >> " + path)
        # print "g"*30
        frames_path = sorted([os.path.join(path, x) for x in os.listdir(path)])
        frames = [ndimage.imread(frame_path) for frame_path in frames_path]
        # print frames
        # print "h"*30
        self.handle_type(frames)
        return self

    def from_video(self, path):
        # print "e"*30
        frames = self.get_video_frames(path)
        # print frames
        # print "f"*30
        self.handle_type(frames)
        return self

    def from_array(self, frames):
        self.handle_type(frames)
        return self

    def handle_type(self, frames):
        if self.vtype == 'mouth':
            self.process_frames_mouth(frames)
        elif self.vtype == 'face':
            self.process_frames_face(frames)
        else:
            raise Exception('Video type not found')

    def process_frames_face(self, frames):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.face_predictor_path)
        mouth_frames = self.get_frames_mouth(detector, predictor, frames)
        self.face = np.array(frames)
        self.mouth = np.array(mouth_frames)
        self.set_data(mouth_frames)

    def process_frames_mouth(self, frames):
        self.face = np.array(frames)
        self.mouth = np.array(frames)
        self.set_data(frames)

    def get_frames_mouth(self, detector, predictor, frames):
        MOUTH_WIDTH = 100
        MOUTH_HEIGHT = 50
        HORIZONTAL_PAD = 0.19
        normalize_ratio = None
        mouth_frames = []
        for frame in frames:
            dets = detector(frame, 1)
            shape = None
            for k, d in enumerate(dets):
                shape = predictor(frame, d)
                i = -1
            if shape is None: # Detector doesn't detect face, just return as is
                return frames
            mouth_points = []
            for part in shape.parts():
                i += 1
                if i < 48: # Only take mouth region
                    continue
                mouth_points.append((part.x,part.y))
            np_mouth_points = np.array(mouth_points)

            mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

            if normalize_ratio is None:
                mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
                mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

                normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

            new_img_shape = (int(frame.shape[0] * normalize_ratio), int(frame.shape[1] * normalize_ratio))
            resized_img = imresize(frame, new_img_shape)

            mouth_centroid_norm = mouth_centroid * normalize_ratio

            mouth_l = int(mouth_centroid_norm[0] - MOUTH_WIDTH / 2)
            mouth_r = int(mouth_centroid_norm[0] + MOUTH_WIDTH / 2)
            mouth_t = int(mouth_centroid_norm[1] - MOUTH_HEIGHT / 2)
            mouth_b = int(mouth_centroid_norm[1] + MOUTH_HEIGHT / 2)

            mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]

            mouth_frames.append(mouth_crop_image)
        return mouth_frames

    def get_video_frames(self, path):
        # print "i"*30
        # print path
        videogen = skvideo.io.vreader(path)

        # print videogen
        # print "j"*30
        frames = np.array([frame for frame in videogen])
        # print "k"*30
        # print frames
        return frames

    def set_data(self, frames):
        data_frames = []
        # print "=====start dataframes====="
        
        
        for frame in frames:
            frame = frame.swapaxes(0,1) # swap width and height to form format W x H x C
            if len(frame.shape) < 3:
                frame = np.array([frame]).swapaxes(0,2).swapaxes(0,1) # Add grayscale channel
            data_frames.append(frame)
        frames_n = len(data_frames)
        data_frames = np.array(data_frames) # T x W x H x C
        if K.image_data_format() == 'channels_first':
            data_frames = np.rollaxis(data_frames, 3) # C x T x W x H
        self.data = data_frames
        self.length = frames_n

        # print data_frames
        # print "=====end dataframes====="

class Align(object):
    def __init__(self, absolute_max_string_len=32, label_func=None):
        self.label_func = label_func
        self.absolute_max_string_len = absolute_max_string_len

    def from_file(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
            print "lines :" + str(lines)
        align = [(int(y[0])/1000, int(y[1])/1000, y[2]) for y in [x.decode('utf-8').strip().split(" ") for x in lines]]
        self.build(align)
        return self

    def from_array(self, align):
        self.build(align)
        return self

    def build(self, align):
        self.align = self.strip(align, ['sp','sil'])
        print self.align
        self.sentence = self.get_sentence(align)
        print self.sentence
        self.label = self.get_label(self.sentence) #text_to_labels
        print self.label
        self.padded_label = self.get_padded_label(self.label)
        print self.padded_label

    def strip(self, align, items):
        return [sub for sub in align if sub[2] not in items]

    def get_sentence(self, align):
        return " ".join([y[-1] for y in align if y[-1] not in ['sp', 'sil']])

    def get_label(self, sentence):
        return self.label_func(sentence)

    def get_padded_label(self, label):        
        padding = np.ones((self.absolute_max_string_len-len(label))) * -1
        labels = np.concatenate((np.array(label), padding), axis=0)
        new_labels = []
        for l in labels:
            new_labels.append(int(l))
        return new_labels
        # return labels

    @property
    def word_length(self):
        return len(self.sentence.split(" "))

    @property
    def sentence_length(self):
        return len(self.sentence)

    @property
    def label_length(self):
        return len(self.label)

def show_video_subtitle(frames, subtitle):
    fig, ax = plt.subplots()
    fig.show()

    plt.rcParams["font.family"] = 'NanumMyeongjo'

    print "++++++++++++++++++++++++++"
    print subtitle
    print "++++++++++++++++++++++++++"

    text = plt.text(0.5, 0.1, "", 
        ha='center', va='center', transform=ax.transAxes, 
        fontdict={'fontsize': 15, 'color':'white', 'fontweight': 500})
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),
        path_effects.Normal()])

    subs = subtitle.split()
    inc = max(len(frames)/(len(subs)+1), 0.01)

    i = 0
    img = None
    for frame in frames:
        sub = " ".join(subs[:int(i/inc)])
        # print frame
        print str(i) + " : " + sub
        # print sub
        text.set_text(sub)

        if img is None:
            img = plt.imshow(frame)
        else:
            img.set_data(frame)
        fig.canvas.draw()
        i += 1

def _decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    """Decodes the output of a softmax.
    Can use either greedy search (also known as best path)
    or a constrained dictionary search.
    # Arguments
        y_pred: tensor `(samples, time_steps, num_categories)`
            containing the prediction, or output of the softmax.
        input_length: tensor `(samples, )` containing the sequence length for
            each batch item in `y_pred`.
        greedy: perform much faster best-path search if `true`.
            This does not use a dictionary.
        beam_width: if `greedy` is `false`: a beam search decoder will be used
            with a beam of this width.
        top_paths: if `greedy` is `false`,
            how many of the most probable paths will be returned.
    # Returns
        Tuple:
            List: if `greedy` is `true`, returns a list of one element that
                contains the decoded sequence.
                If `false`, returns the `top_paths` most probableo
                decoded sequences.
                Important: blank labels are returned as `-1`.
            Tensor `(top_paths, )` that contains
                the log probability of each decoded sequence.
    """
    decoded = K.ctc_decode(y_pred=y_pred, input_length=input_length,
                           greedy=greedy, beam_width=beam_width, top_paths=top_paths)
    #print y_pred

    print "path.eval+++++"
    for path in decoded[0]:
        print path.eval(session=K.get_session())

    print "+++++path.eval"

    # print "&&&&&path&&&&&"
    paths = [path.eval(session=K.get_session()) for path in decoded[0]]
    # print "&&&&&"

    print "logprobs$$$$$$$$$$$$$"
    logprobs  = decoded[1].eval(session=K.get_session())
    print logprobs
    return (paths, logprobs)

def decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1, **kwargs):
    language_model = kwargs.get('language_model', None)

    print "@@@@@@@@@@@@@decode@@@@@@@@@@@@@"
    print y_pred
    print "######################"
    print input_length
    print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

    paths, logprobs = _decode(y_pred=y_pred, input_length=input_length,
                              greedy=greedy, beam_width=beam_width, top_paths=top_paths)
    print "paths : " + str(paths)
    print "log : " + str(logprobs)

    if language_model is not None:
        # TODO: compute using language model
        raise NotImplementedError("Language model search is not implemented yet")
    else:
        # simply output highest probability sequence
        # paths has been sorted from the start
        result = paths[0]
    return result

class Decoder(object):
    def __init__(self, greedy=True, beam_width=100, top_paths=1, **kwargs):
        self.greedy         = greedy
        self.beam_width     = beam_width
        self.top_paths      = top_paths
        self.language_model = kwargs.get('language_model', None)
        self.postprocessors = kwargs.get('postprocessors', []) #labels_to_text, spell.sentence

    def decode(self, y_pred, input_length):
        decoded = decode(y_pred, input_length, greedy=self.greedy, beam_width=self.beam_width,
                         top_paths=self.top_paths, language_model=self.language_model)
        preprocessed = []

        for output in decoded: #decoded 한번 돔(아마 여기서 여러번 돌아야 여러 문장이 가능)            
            out = output

            for postprocessor in self.postprocessors: # 2번 실행 labels_to_text -> spell.sentence
                out = postprocessor(out)
            preprocessed.append(out)

        return preprocessed

def labels_to_text(labels):
    # 26 is space, 가 = 44032 / 힣 = 55203
    text = ''
    for c in labels:
        if c >= 0 and c <= 11171:
            c += ord(u'가')
            text += unichr(c)
        elif c == 11172:
            text += ' '
    return text

# def text_to_labels(text):
#     ret = []
#     for char in text:
#         if char >= 'a' and char <= 'z':
#             ret.append(ord(char) - ord('a'))
#         elif char == ' ':
#             ret.append(26)
#     return ret

# def labels_to_text(labels):
#     # 26 is space, 27 is CTC blank char
#     text = ''
#     for c in labels:
#         if c >= 0 and c < 26:
#             text += chr(c + ord('a'))
#         elif c == 26:
#             text += ' '
#     return text

class Spell(object):
    def __init__(self, path): # 

        self.dictionary = Counter(self.words(io.open(path, 'r', encoding='utf-8').read()))
        print "dictionary" + str(self.dictionary)

    def words(self, text):
        text = text.replace('\n', ' ')
        hangul = re.findall('[^ ㄱ-ㅣ가-힣]+', text)

        hangulencode = []
        for i in range(len(hangul)):
            hangulencode.append(hangul[i])
            #print hangul[i]
        #print hangulencode
        return hangulencode

    def P(self, word, N=None):
        "Probability of `word`."
        if N is None:
            N = sum(self.dictionary.values())
        return self.dictionary[word] / N

    def correction(self, word):
        "Most probable spelling correction for word."
        return max(self.candidates(word), key=self.P)

    def candidates(self, word):
        "Generate possible spelling corrections for word."
        return (self.known([word]) or self.known([word]) or [word] or [word])

    def known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.dictionary)

    # def edits1(self, word):
    #     "All edits that are one edit away from `word`."
    #     letters    = 'abcdefghijklmnopqrstuvwxyz'
    #     splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    #     deletes    = [L + R[1:]               for L, R in splits if R]
    #     transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    #     replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    #     inserts    = [L + c + R               for L, R in splits for c in letters]
    #     return set(deletes + transposes + replaces + inserts)

    # def edits2(self, word):
    #     "All edits that are two edits away from `word`."
    #     return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    # Correct words
    def corrections(self, words):
        return [self.correction(word) for word in words]

    # Correct sentence
    def sentence(self, sentence):
        #return untokenize(sentence)

        print "sentence==="
        print sentence
        print "unicode_sentence==="
        print unicode(sentence)

        return unicode(sentence)

class LipNet(object):
    def __init__(self, img_c=3, img_w=100, img_h=50, frames_n=75, absolute_max_string_len=32, output_size=11174):
        self.img_c = img_c
        self.img_w = img_w
        self.img_h = img_h
        self.frames_n = frames_n
        self.absolute_max_string_len = absolute_max_string_len
        self.output_size = output_size
        self.build() 

    def build(self):
        if K.image_data_format() == 'channels_first':
            input_shape = (self.img_c, self.frames_n, self.img_w, self.img_h)
        else:
            input_shape = (self.frames_n, self.img_w, self.img_h, self.img_c)

        self.input_data = Input(name='the_input', shape=input_shape, dtype='float32')

        self.zero1 = ZeroPadding3D(padding=(1, 2, 2), name='zero1')(self.input_data)
        self.conv1 = Conv3D(32, (3, 5, 5), strides=(1, 2, 2), kernel_initializer='he_normal', name='conv1')(self.zero1)
        self.batc1 = BatchNormalization(name='batc1')(self.conv1)
        self.actv1 = Activation('relu', name='actv1')(self.batc1)
        self.drop1 = SpatialDropout3D(0.5)(self.actv1)
        self.maxp1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(self.drop1)

        self.zero2 = ZeroPadding3D(padding=(1, 2, 2), name='zero2')(self.maxp1)
        self.conv2 = Conv3D(64, (3, 5, 5), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv2')(self.zero2)
        self.batc2 = BatchNormalization(name='batc2')(self.conv2)
        self.actv2 = Activation('relu', name='actv2')(self.batc2)
        self.drop2 = SpatialDropout3D(0.5)(self.actv2)
        self.maxp2 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2')(self.drop2)

        self.zero3 = ZeroPadding3D(padding=(1, 1, 1), name='zero3')(self.maxp2)
        self.conv3 = Conv3D(96, (3, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv3')(self.zero3)
        self.batc3 = BatchNormalization(name='batc3')(self.conv3)
        self.actv3 = Activation('relu', name='actv3')(self.batc3)
        self.drop3 = SpatialDropout3D(0.5)(self.actv3)
        self.maxp3 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3')(self.drop3)

        self.resh1 = TimeDistributed(Flatten())(self.maxp3)

        self.gru_1 = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru1'), merge_mode='concat')(self.resh1)
        self.gru_2 = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru2'), merge_mode='concat')(self.gru_1)

        # transforms RNN output to character activations:
        self.dense1 = Dense(self.output_size, kernel_initializer='he_normal', name='dense1')(self.gru_2)

        self.y_pred = Activation('softmax', name='softmax')(self.dense1)

        self.labels = Input(name='the_labels', shape=[self.absolute_max_string_len], dtype='float32')
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')
        self.loss_out = CTC('ctc', [self.y_pred, self.labels, self.input_length, self.label_length])

        self.model = Model(inputs=[self.input_data, self.labels, self.input_length, self.label_length], outputs=self.loss_out)

    def summary(self):
        Model(inputs=self.input_data, outputs=self.y_pred).summary()

    def predict(self, input_batch):
        return self.test_function([input_batch, 0])[0]  # the first 0 indicates test

    @property
    def test_function(self):
        # captures output of softmax so we can decode the output during visualization
        return K.function([self.input_data, K.learning_phase()], [self.y_pred, K.learning_phase()])

def CTC(name, args):
	return Lambda(ctc_lambda_func, output_shape=(1,), name=name)(args)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # From Keras example image_ocr.py:
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    #y_pred = y_pred[:, 2:, :]
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


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

    print(result)
    if(result == u'\uae30'):
        result = u'\uae30\uc5ed'
    elif(result == u'\ub2c8'):
        result = u'\ub2c8\uc740'
    elif(result == u'\ub514'):
        result = u'\ub514\uadff'
    elif(result == u'\ub9ac'):
        result = u'\ub9ac\uc744'
    elif(result == u'\ubbf8'):
        result = u'\ubbf8\uc74c'
    elif(result == u'\ube44'):
        result = u'\ube44\uc74d'
    elif(result == u'\uc2dc'):
        result = u'\uc2dc\uc637'
    elif(result == u'\uc774'):
        result = u'\uc774\uc751'
    elif(result == u'\uc9c0'):
        result = u'\uc9c0\uc752'
    elif(result == u'\uce58'):
        result = u'\uce58\uc753'
    elif(result == u'\ud0a4'):
        result = u'\ud0a4\uc754'
    elif(result == u'\ud2f0'):
        result = u'\ud2f0\uc755'
    elif(result == u'\ud53c'):
        result = u'\ud53c\uc756'
    elif(result == u'\ud788'):
        result = u'\ud788\uc757'
    print(" -> " + result)

    return (video, result)

if __name__ == '__main__':
    if len(sys.argv) == 3:
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

