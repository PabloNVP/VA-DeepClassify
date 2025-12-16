"""
Script para aumentar el dataset con transformaciones de imágenes.
Genera múltiples variaciones de cada imagen original.
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Directorio de origen y destino
SOURCE_DIR = "./data/train"
DEST_DIR = "./data_augmented/train"
AUGMENTATIONS_PER_IMAGE = 5  # Número de imágenes aumentadas por imagen original


def augment_dataset(source_dir=SOURCE_DIR, dest_dir=DEST_DIR, 
                    augmentations_per_image=AUGMENTATIONS_PER_IMAGE):
    """Genera imágenes aumentadas a partir del dataset original"""
    
    # Crear el directorio de destino si no existe
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Configuración del aumento de datos
    data_gen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Procesar cada clase en el directorio de origen
    for class_name in os.listdir(source_dir):
        class_source_dir = os.path.join(source_dir, class_name)
        class_dest_dir = os.path.join(dest_dir, class_name)

        # Crear el directorio de la clase en el destino
        if not os.path.exists(class_dest_dir):
            os.makedirs(class_dest_dir)

        # Verificar que sea un directorio
        if os.path.isdir(class_source_dir):
            for img_name in os.listdir(class_source_dir):
                img_path = os.path.join(class_source_dir, img_name)

                # Leer la imagen
                img = load_img(img_path)
                img_array = img_to_array(img)
                img_array = img_array.reshape((1,) + img_array.shape)

                # Generar imágenes aumentadas
                i = 0
                for batch in data_gen.flow(img_array, batch_size=1, 
                                           save_to_dir=class_dest_dir, 
                                           save_prefix=f"{class_name}", 
                                           save_format="jpeg"):
                    i += 1
                    if i >= augmentations_per_image:
                        break

    print(f"Aumento de datos completado. Las imágenes aumentadas se guardaron en: {dest_dir}")


if __name__ == "__main__":
    augment_dataset()
