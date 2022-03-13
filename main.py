import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import PIL
import os

image_res = [1920, 1080]
input_filetype = '.png'
input_directory = './input/'
output_directory = './output/'
style_location = './style.jpg'

def load_image(image_path, image_size=(image_res[0], image_res[1])):
  img = tf.io.decode_image(
    tf.io.read_file(image_path),
    channels=3, dtype=tf.float32)[tf.newaxis, ...]
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

def export_image(tf_img):
    tf_img = tf_img*255
    tf_img = np.array(tf_img, dtype=np.uint8)
    if np.ndim(tf_img)>3:
        assert tf_img.shape[0] == 1
        img = tf_img[0]
    return PIL.Image.fromarray(img)


directory = os.fsencode(input_directory)
style_image = load_image(style_location)
stylize_model = tf_hub.load('tf_model')

def cycleInputFolder():
  for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(input_filetype):
      print("Processing: " + filename)
      original_image = load_image(input_directory + filename)
      results = stylize_model(tf.constant(original_image), tf.constant(style_image))
      stylized_photo = results[0]
      export_image(stylized_photo).save(output_directory + filename)


if __name__ == "__main__":
  cycleInputFolder()