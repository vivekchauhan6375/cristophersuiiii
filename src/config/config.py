import pathlib
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop

import src

training_data = None
X_train = None
Y_train = None

epochs = 100
mb_size = 2

try:
    PACKAGE_ROOT = pathlib.Path(src.__file__).resolve().parent
except AttributeError:
    PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent  
    print("Warning: 'src' module not found or '__file__' attribute is missing. Using fallback PACKAGE_ROOT.")

DATAPATH = os.path.join(PACKAGE_ROOT, "datasets")
SAVED_MODEL_PATH = os.path.join(PACKAGE_ROOT, "trained_models")
