""" Módulo de configuración del proyecto """

import configparser
import os

# Ruta al archivo de configuración
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.ini")

# Leer configuración
_config = configparser.ConfigParser()
_config.read(CONFIG_PATH)

# ============ DIRECTORIOS DE DATOS ============
SOURCE_DIR = _config.get("directories", "source_dir")
DATA_DIR = _config.get("directories", "data_dir")
TRAIN_DIR = _config.get("directories", "train_dir")
VAL_DIR = _config.get("directories", "val_dir")
TEST_DIR = _config.get("directories", "test_dir")
MODEL_SAVE_DIR = _config.get("directories", "model_save_dir")

# ============ DIVISIÓN DEL DATASET ============
TRAIN_SPLIT = _config.getfloat("dataset_split", "train_split")
VAL_SPLIT = _config.getfloat("dataset_split", "val_split")
TEST_SPLIT = _config.getfloat("dataset_split", "test_split")

# ============ CONFIGURACIÓN DEL MODELO ============
IMG_SIZE = (
    _config.getint("model", "img_width"),
    _config.getint("model", "img_height")
)
NUM_CLASSES = _config.getint("model", "num_classes")
MODEL_ARCHITECTURE = _config.get("model", "architecture")

# ============ CONFIGURACIÓN DE ENTRENAMIENTO ============
BATCH_SIZE = _config.getint("training", "batch_size")
EPOCHS = _config.getint("training", "epochs")
LEARNING_RATE = _config.getfloat("training", "learning_rate")

# ============ ARQUITECTURA ============
DENSE_UNITS = _config.getint("architecture", "dense_units")
DROPOUT_RATE = _config.getfloat("architecture", "dropout_rate")
