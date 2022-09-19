import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


#  2 : tensorflow version
import tensorflow as tf
#print(tf.__version__)


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())



