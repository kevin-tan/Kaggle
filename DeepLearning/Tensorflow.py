import numpy as np
from os.path import join
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import img_to_array, load_img
from tensorflow.python.keras.applications import ResNet50
from tools.utils.decode_predictions import decode_predictions


def resNet50(img_paths, img_size):
    def read_and_prep_images(img_paths, img_height=img_size, img_width=img_size):
        imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
        img_array = np.array([img_to_array(image) for image in imgs])
        return preprocess_input(img_array)

    model = ResNet50(weights="../tools/resnet50/weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5")
    test_data = read_and_prep_images(img_paths)
    preds = model.predict(test_data)
    most_likely = decode_predictions(preds, top=3,
                                     class_list_path='../tools/resnet50/imagenet_class_index.json')
    for i, img_paths in enumerate(img_paths):
        print(most_likely[i])


def runDogBreedPredictorWithResnet50():
    image_dir = "../data/dogbreed/train"
    img_paths = [join(image_dir, filename) for filename in ['0a0c223352985ec154fd604d7ddceabd.jpg',
                                                            '0a1b0b7df2918d543347050ad8b16051.jpg',
                                                            '0a1f8334a9f583cac009dc033c681e47.jpg',
                                                            '0a001d75def0b4352ebde8d07c0850ae.jpg',
                                                            '0a3f1898556115d6d0931294876cd1d9.jpg']]
    resNet50(img_paths, 224)



def runHotDogPrefictorWithResnet50():
    hot_dog_image_dir = "../data/seefood/train/hot_dog"
    not_hot_dog_image_dir = "../data/seefood/train/not_hot_dog"
    hot_dog_paths = [join(hot_dog_image_dir, img) for img in ['116486.jpg',
                                                              '7896.jpg']]
    not_hot_dog_paths = [join(not_hot_dog_image_dir, img) for img in ['4781.jpg',
                                                                      '90167.jpg']]
    image_paths = hot_dog_paths + not_hot_dog_paths
    resNet50(image_paths, 224)



runHotDogPrefictorWithResnet50()
