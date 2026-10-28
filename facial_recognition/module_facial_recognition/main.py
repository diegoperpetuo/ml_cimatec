import os
import sys
from pathlib import Path

# Configura encoding para UTF-8 para suportar caracteres Unicode e emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
os.environ["PYTHONIOENCODING"] = "utf-8"

# Garante que o diretório raiz de facial_recognition esteja no sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2
import module_facial_recognition.config as config
from src.camera import CameraManager
from src.detector import FaceRecognizer

def main():
    camera = CameraManager()
    recognizer = FaceRecognizer()

    counter = 0
    current_name = "Analisando..."

    print("Iniciando reconhecimento. Pressione 'q' para sair.")

    while True:
        ret, frame = camera.read_frame()
        if not ret:
            print("Erro ao capturar imagem da câmera.")
            break

        # Processamento periódico do DeepFace
        if counter % config.CHECK_EVERY_N_FRAMES == 0:
            current_name = recognizer.identify(frame)

        counter += 1

        # Renderização
        frame_drawn = camera.draw_info(frame, current_name)
        camera.show("Reconhecimento Facial Modular", frame_drawn)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()

if __name__ == "__main__":
    main()