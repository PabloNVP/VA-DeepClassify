"""
Script para clasificación en tiempo real usando la cámara.
Carga un modelo entrenado y muestra las predicciones en vivo.
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Configuración
# Obtener la ruta base del proyecto (dos niveles arriba de este archivo)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "efficientnet_carne_vacuna.h5")
IMG_SIZE = (224, 224)
CLASS_NAMES = ["asado", "entrania", "matambre", "nalga", "paleta", "vacio"]


def run_realtime_classification(model_path=DEFAULT_MODEL_PATH, class_names=CLASS_NAMES):
    """Ejecuta la clasificación en tiempo real usando la cámara"""
    
    print(f"Cargando modelo desde {model_path}...")
    model = load_model(model_path)
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

        # Preprocesamiento
        img = cv2.resize(frame, IMG_SIZE)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # Predicción
        predictions = model.predict(img)
        class_idx = np.argmax(predictions)
        class_name = class_names[class_idx]
        confidence = predictions[0][class_idx]

        # Mostrar resultados
        text = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Clasificacion en tiempo real", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_classification()
