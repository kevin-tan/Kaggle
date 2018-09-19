# Without the __init__.py file, Python will still search the directory to find modules to import from the package
# i.e. from directory import fileName

# If we do add an __init__.py like this and add the modules part of the directory,
# we can import the package and access each module through one import
# e.g. import DeepLearning as dl; dl.Tensorflow.resNet50(); dl.MNISTModelFromScratch.digitsPredict();

from . import MNISTModelFromScratch
from . import Tensorflow
