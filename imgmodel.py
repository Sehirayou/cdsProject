import streamlit as st

import numpy as np
import pandas as pd
import os, random, glob
# import cv2
import re
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')
from PIL import Image


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

from tensorflow.keras import models
from tensorflow.keras.utils import plot_model

import numpy as np
import time

st.title("Loading the model")

model = load_model(".vscode/model/model_6.keras")

st.markdown(model.summary())
