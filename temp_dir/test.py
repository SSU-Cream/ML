from flask import Flask
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

@app.route('/')
def hello_world():
    return "Hello World!"

app.run()
