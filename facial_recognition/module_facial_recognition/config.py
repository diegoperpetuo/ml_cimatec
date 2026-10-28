import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_PATH = os.path.join(BASE_DIR, "data", "images")

CAMERA_INDEX = 0
CHECK_EVERY_N_FRAMES = 30
CHECK_FRAMES = CHECK_EVERY_N_FRAMES

# Modelo DeepFace
MODEL_NAME = "VGG-Face"