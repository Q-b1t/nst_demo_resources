import numpy as np
import tensorflow as tf

# compute the gram matrix of the features
def gram_matrix(x):
  #x = tf.convert_to_tensor(x, tf.int32)
  x = tf.transpose(x,perm = (2,0,1))
  features = tf.reshape(x,(tf.shape(x)[0],-1))
  gram = tf.matmul(features,tf.transpose(features))
  return gram

# compute the style cost function
def style_cost_function(style_image,generated_image,rows_cols):
  img_nrows,img_ncols = rows_cols
  S = gram_matrix(style_image)
  C = gram_matrix(generated_image)
  channels = 3
  size = img_nrows * img_ncols
  return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

# compute the content cost function
def content_cost_function(base_image,generated_image):
  return tf.reduce_sum(tf.square(tf.subtract(generated_image,base_image)))

def loss_function(
    generated_image, 
    base_image, 
    style_image,
    content_layer,
    style_layers,
    feature_extractor,
    weights,
    rows_cols):
  
  # fetch the weights
  content_weight,style_weight = weights

  # combine the images into a single tensor
  input_tensor = tf.concat(
      [base_image, style_image, generated_image], axis=0
  )

  # get the values in all the layers for the three images
  features = feature_extractor(input_tensor)

  # iinitialize the loss function
  loss = tf.zeros(shape=())

  # compute the content loss
  layer_features = features[content_layer]
  base_image_features = layer_features[0, :, :, :]
  combination_features = layer_features[2, :, :, :]

  loss = loss + content_weight * content_cost_function(
      base_image_features, combination_features
  )

  # compute the style loss
  for layer_name in style_layers:
      layer_features = features[layer_name]
      style_reference_features = layer_features[1, :, :, :]
      combination_features = layer_features[2, :, :, :]
      sl = style_cost_function(style_reference_features, combination_features,rows_cols)
      loss += (style_weight / len(style_layers)) * sl

  return loss


