import cv2
import module_facial_recognition.config as config

class CameraManager:
    def __init__(self, source=config.CAMERA_INDEX):
        self.cap = cv2.VideoCapture(source)

    def read_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def draw_info(self, frame, name):
        """Desenha o nome da pessoa reconhecida sobre o frame."""
        cv2.putText(
            frame,
            f"Pessoa: {name}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        return frame

    def show(self, window_name, frame):
        cv2.imshow(window_name, frame)

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()