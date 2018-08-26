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


# ---------------------------------------- Transfer learning ---------------------------------------- #

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

# Dense is the new prediction layers
# Sequential is the main layer that will contain the pre-trained model

# Number of nodes in the prediction layer, i.e. what are the categories we're trying to predict to in this case 2 Urban vs Rural
num_classes = 2
# Without top means without the prediction layer
resnet_weight = '../tools/resnet50/weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

# -- Setup of new model --
new_model = Sequential()
# Pooling = avg means if we have extra channels in our tensor at the end of this step collapse them into a 1D tensor by taking avg of all channels
new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weight))
# Apply a softmax function to turn the results into probability
new_model.add(Dense(num_classes, activation='softmax'))
# Tell tensor not to train first layer since we only want to train the new prediction layer
new_model.layers[0].trainable = False
# Optimizing using stochastic gradient descent for minimize categorical_crossentropy and we want the results in accuracy
new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# -- Loading and training new model --

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224
# Apply preprocess_input to every image found when generating the img set
data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
# Load the dataset according to the packages
train_gen = data_gen.flow_from_directory('../data/urban_rural/train', target_size=(image_size, image_size),
                                         batch_size=24, class_mode="categorical")
test_gen = data_gen.flow_from_directory('../data/urban_rural/test', target_size=(image_size, image_size),
                                        class_mode="categorical")

# Train
# new_model.fit_generator(train_gen, steps_per_epoch=3, validation_data=test_gen, validation_steps=1)

# Data augmentation to maximize training data set
# X is the images within the subdirectories combined, y is the labels of subdirectories
data_gen_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True, width_shift_range=0.2, height_shift_range=0.2)
train_gen_aug = data_gen_with_aug.flow_from_directory('../data/urban_rural/train', target_size=(image_size, image_size),
                                         batch_size=12, class_mode="categorical")
new_model.fit_generator(train_gen_aug, steps_per_epoch=6, epochs=2, validation_data=test_gen, validation_steps=1)
