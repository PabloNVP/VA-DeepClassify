""" Script para dividir y eliminar duplicados del dataset en conjuntos de entrenamiento, validación y prueba."""

import os
import shutil
import random
import hashlib

# Importar configuración desde archivo centralizado
from src.config import (
    SOURCE_DIR, DATA_DIR,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
)


def calculate_image_hash(image_path):
    """Función para calcular el hash de una imagen"""
    with open(image_path, "rb") as f:
        img_hash = hashlib.md5(f.read()).hexdigest()
    return img_hash


def split_dataset(source_dir=SOURCE_DIR, dest_dir=DATA_DIR, 
                  train_split=TRAIN_SPLIT, val_split=VAL_SPLIT):
    """Divide el dataset en train, val y test eliminando duplicados"""
    
    # Crear carpetas de destino
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(dest_dir, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)

    # Almacenar hashes para evitar duplicados
    image_hashes = set()

    # Recorrer las clases en la carpeta de origen
    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        if os.path.isdir(class_dir):
            # Crear carpetas para la clase en train, val y test
            for split in ["train", "val", "test"]:
                split_class_dir = os.path.join(dest_dir, split, class_name)
                if not os.path.exists(split_class_dir):
                    os.makedirs(split_class_dir)

            # Obtener todas las imágenes de la clase
            images = os.listdir(class_dir)
            random.shuffle(images)  # Mezclar las imágenes aleatoriamente

            # Filtrar imágenes duplicadas
            unique_images = []
            for img_name in images:
                img_path = os.path.join(class_dir, img_name)
                img_hash = calculate_image_hash(img_path)
                if img_hash not in image_hashes:
                    image_hashes.add(img_hash)
                    unique_images.append(img_name)

            # Calcular cantidades para cada conjunto
            total_images = len(unique_images)
            train_count = int(total_images * train_split)
            val_count = int(total_images * val_split)

            # Dividir las imágenes
            train_images = unique_images[:train_count]
            val_images = unique_images[train_count:train_count + val_count]
            test_images = unique_images[train_count + val_count:]

            # Copiar las imágenes a las carpetas correspondientes
            for img_name in train_images:
                shutil.copy(os.path.join(class_dir, img_name), 
                           os.path.join(dest_dir, "train", class_name, img_name))
            for img_name in val_images:
                shutil.copy(os.path.join(class_dir, img_name), 
                           os.path.join(dest_dir, "val", class_name, img_name))
            for img_name in test_images:
                shutil.copy(os.path.join(class_dir, img_name), 
                           os.path.join(dest_dir, "test", class_name, img_name))

    print("Preprocesamiento completado. Las imágenes únicas se han distribuido en las carpetas de train, val y test.")


if __name__ == "__main__":
    split_dataset()
