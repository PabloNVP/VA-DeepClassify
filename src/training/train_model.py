""" Script de entrenamiento genérico para clasificación de imágenes. (ImageNet) """

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50, DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# Importar configuración desde archivo centralizado
from src.config import (
    IMG_SIZE, BATCH_SIZE, EPOCHS, NUM_CLASSES, LEARNING_RATE,
    DENSE_UNITS, DROPOUT_RATE, MODEL_ARCHITECTURE,
    TRAIN_DIR, VAL_DIR, TEST_DIR, MODEL_SAVE_DIR
)

# Importar registro de arquitecturas
from src.training.architectures import ARCHITECTURES

# Mapeo de arquitecturas a clases de Keras (lazy loading)
MODEL_CLASSES = {
    "efficientnet": EfficientNetB0,
    "mobilenet": MobileNetV2,
    "resnet": ResNet50,
    "densenet": DenseNet121
}


def get_model_save_path(architecture: str) -> str:
    """Genera la ruta de guardado según la arquitectura"""
    return os.path.join(MODEL_SAVE_DIR, f"{architecture}_carne_vacuna.h5")


def create_data_generators(train_dir=TRAIN_DIR, val_dir=VAL_DIR, test_dir=TEST_DIR):
    """Crea los generadores de datos para entrenamiento, validación y prueba"""
    
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )
    
    return train_generator, val_generator, test_generator


def create_model(architecture: str = MODEL_ARCHITECTURE, num_classes: int = NUM_CLASSES):
    """
    Crea el modelo con transfer learning según la arquitectura especificada.
    
    Args:
        architecture: Nombre de la arquitectura ("efficientnet", "mobilenet")
        num_classes: Número de clases para clasificación
    
    Returns:
        Modelo compilado listo para entrenar
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Arquitectura '{architecture}' no soportada. "
            f"Opciones: {list(ARCHITECTURES.keys())}"
        )
    
    arch_name = ARCHITECTURES[architecture]["name"]
    model_class = MODEL_CLASSES[architecture]
    
    print(f"Usando arquitectura: {arch_name}")
    
    # Crear modelo base con pesos de ImageNet
    base_model = model_class(
        weights="imagenet", 
        include_top=False, 
        input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = False

    # Añadir capas de clasificación
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(DENSE_UNITS, activation="relu")(x)
    x = Dropout(DROPOUT_RATE)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model


def train_model(architecture: str = MODEL_ARCHITECTURE):
    """
    Entrena el modelo con la arquitectura especificada.
    
    Args:
        architecture: Nombre de la arquitectura a usar
    """
    print(f"\n{'='*50}")
    print(f"Iniciando entrenamiento con {ARCHITECTURES[architecture]['name']}")
    print(f"{'='*50}\n")
    
    print("Creando generadores de datos...")
    train_gen, val_gen, test_gen = create_data_generators()
    
    print("Creando modelo...")
    model = create_model(architecture=architecture)
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("Iniciando entrenamiento...")
    model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen
    )

    print("Evaluando modelo...")
    eval_results = model.evaluate(test_gen)
    print(f"Test Accuracy: {eval_results[1] * 100:.2f}%")

    model_save_path = get_model_save_path(architecture)
    print(f"Guardando modelo en {model_save_path}...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print("Entrenamiento completado.")
    
    return model