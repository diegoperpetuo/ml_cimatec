import os
from deepface import DeepFace
import module_facial_recognition.config as config

class FaceRecognizer:
    def __init__(self, db_path=config.DB_PATH, model_name=config.MODEL_NAME):
        self.db_path = db_path
        self.model_name = model_name

    def identify(self, frame):
        """Identifica um rosto no frame informado retornando o nome da pessoa."""
        try:
            results = DeepFace.find(
                img_path=frame,
                db_path=self.db_path,
                model_name=self.model_name,
                enforce_detection=False,
                silent=True
            )

            if len(results) > 0 and not results[0].empty:
                matched_path = results[0]['identity'][0]
                name = os.path.splitext(os.path.basename(matched_path))[0]
                return name
            return "Desconhecido"
        except Exception:
            return "Desconhecido"