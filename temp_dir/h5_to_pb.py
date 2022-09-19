from tensorflow import keras

model = keras.models.load_model('./model/weights114.h5', compile=False)
#model = keras.models.load_weights('./model/my_model.h5', compile=False)

export_path = './pb'
model.save(export_path, save_format="tf")
