import tensorflow as tf

def preprocess (img_path):

    # Read in image from file path
    byte_img = tf.io.read_file(img_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)

    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (105,105))
    # Scale image to be between 0 and 1
    img = img/255


    # Return image
    return img
