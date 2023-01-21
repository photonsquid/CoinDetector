from src.dataset import create_pairs
import os
import random

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

KERNEL_LOCATION = "local"                # TODO: auto-detect kernel location
LOG_LEVEL = "info"                       # TODO: handle log level
DATASET_SOURCE = "huggingface"
DATASET_NAME = "photonsquid/coins-euro"

# Import standard dependencies
