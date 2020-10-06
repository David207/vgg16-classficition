from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from IPython.display import Image

# load an image from file
image = load_img('bmw.jpg', target_size=(224, 224))

# convert the image to array
image = img_to_array(image)

# reshape data
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# prepare the image to vgg16
image = preprocess_input(image)

# load model
model = VGG16()

# predict
hat = model.predict(image)

# convert prob to class labels
label = decode_predictions(hat)

# retrieve the most likely
label = label[0][0]

# print classification
print('%s (%.2f%%)' % (label[1], label[2] * 100))
Image('bmw.jpg')
Image('bmw.jpg', width=224, height=224)
