from insightface.app import FaceAnalysis

class FaceDetection:
    def __init__(self, model_name):
        self.face_app = FaceAnalysis(name=model_name)
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

    def get_face_embedding(self, image):
        faces = self.face_app.get(image)
        if faces:
            return faces[0].embedding
        return None
