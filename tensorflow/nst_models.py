from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19

# instances a pretrained vgg19 model and creates the feature extractor
def get_pretrained_vgg_model_fe():
  # instance the pretrained model
  pretrained_vgg_model = VGG19(include_top = False, weights="imagenet")
  # create a dictionary that maps the layers name to each feature extractor 
  model_outputs = {layer.name : layer.output for layer in pretrained_vgg_model.layers}
  # feature extractor
  feature_extractor = Model(inputs=pretrained_vgg_model.inputs, outputs=model_outputs)
  return pretrained_vgg_model,feature_extractor

# get a list of the layers that will be used to style and content
def get_layers_lists():
  # define the style layers
  style_layers = [
      "block1_conv1",
      "block2_conv1",
      "block3_conv1",
      "block4_conv1",
      "block5_conv1",
      ]
  # define the content layer
  content_layer = "block5_conv2"
  return content_layer,style_layers

# get the weights for each of the cost functions in order to computer the final cost
def get_weights():
  # define the style and content weights
  content_weight = 2.5e-8
  style_weight = 1.0e-6
  return content_weight,style_weight
