import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img,img_to_array

def preprocess_image(image_path,rows_cols):
  img_nrows, img_ncols = rows_cols
  # load the image into a tensot
  img = load_img(image_path, target_size=(img_nrows, img_ncols))
  # turn the image into a numpy array
  img = img_to_array(img)
  # add a batch dimention
  img = np.expand_dims(img, axis=0)
  # preprocess according to the vgg model's specification
  img = preprocess_input(img)
  return tf.convert_to_tensor(img)

def deprocess_image(x,rows_cols):
    img_nrows, img_ncols = rows_cols
    # Cconert to array
    x = x.reshape((img_nrows, img_ncols, 3))
    # mean = 0
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # convert to rgb
    x = x[:, :, ::-1]
    # normalize
    x = np.clip(x, 0, 255).astype("uint8")
    return x
