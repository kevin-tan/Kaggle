import numpy as np
from os.path import join
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import img_to_array, load_img
from tensorflow.python.keras.applications import ResNet50
from tools.utils.decode_predictions import  decode_predictions
from IPython.display import Image, display

image_dir = "../data/dogbreed/train"
img_paths = [join(image_dir, filename) for filename in ['0a0c223352985ec154fd604d7ddceabd.jpg',
                                                        '0a1b0b7df2918d543347050ad8b16051.jpg',
                                                        '0a1f8334a9f583cac009dc033c681e47.jpg',
                                                        '0a001d75def0b4352ebde8d07c0850ae.jpg',
                                                        '0a3f1898556115d6d0931294876cd1d9.jpg']]

image_size = 224


def read_and_prp_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_height)) for img_path in img_paths]
    img_array = np.array([img_to_array(image) for image in imgs])
    return preprocess_input(img_array)

model = ResNet50(weights="../tools/resnet50/weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5")
test_data = read_and_prp_images(img_paths)
preds = model.predict(test_data)

most_likely = decode_predictions(preds, top=3, class_list_path='../tools/resnet50/weights/imagenet_class_index.json')
for i in enumerate(img_paths):
    print(most_likely[i])
