"""
Script para clasificación en tiempo real usando MobileNetV2.
Incluye visualización de FPS y mejoras en la interfaz.
"""

import cv2
import numpy as np
import tensorflow as tf
import os

# Configuración
# Obtener la ruta base del proyecto (dos niveles arriba de este archivo)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
MODEL_PATH = os.path.join(BASE_DIR, "models", "mobilenetv2_finetuned_model.h5")
IMG_SIZE = (224, 224)
CLASS_NAMES = ["asado", "chorizo", "entrania", "matambre", "nalga", "paleta", "vacio"]


def run_mobilenet_classification(model_path=MODEL_PATH, class_names=CLASS_NAMES):
    """Ejecuta la clasificación en tiempo real con MobileNetV2"""
    
    print("Cargando modelo...")
    model = tf.keras.models.load_model(model_path)
    print("Modelo cargado correctamente.")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: no se pudo abrir la cámara.")
        return

    print("Presiona 'q' para salir del programa.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()

        # Preprocesamiento
        img = cv2.resize(frame, IMG_SIZE)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # Predicción
        preds = model.predict(img)
        class_id = np.argmax(preds)
        class_name = class_names[class_id]
        confidence = preds[0][class_id]

        # Dibujar resultados
        text = f"{class_name} ({confidence*100:.1f}%)"
        cv2.putText(display_frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Clasificacion en tiempo real", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_mobilenet_classification()
