import cv2
from scipy.spatial.distance import cosine
from utils.face_detection import FaceDetection
from ultralytics import YOLO

class PersonTracker:
    SIMILARITY_THRESHOLD = 0.5

    def __init__(self, yolo_model_path, face_model_name, mongo_handler, use_case, camera_source):
        # Load YOLO model
        self.yolo_model = YOLO(yolo_model_path)

        # Load Face Detection
        self.face_detection = FaceDetection(face_model_name)

        # MongoDB handler and parameters
        self.mongo_handler = mongo_handler
        self.use_case = use_case
        self.camera_source = camera_source

        # Person database
        self.person_database = {}
        self.next_person_id = 1

    def is_new_person(self, new_embedding):
        for person_id, stored_embedding in self.person_database.items():
            similarity = 1 - cosine(new_embedding, stored_embedding)
            if similarity > self.SIMILARITY_THRESHOLD:
                return person_id  # Existing person detected

        # If no match, assign new ID
        self.person_database[self.next_person_id] = new_embedding
        self.next_person_id += 1
        return self.next_person_id - 1

    def process_frame(self, frame):
        # Run YOLO for person detection
        results = self.yolo_model(frame)

        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box.tolist()
                if int(cls) == 0:  # Class 0 corresponds to 'person'
                    person_crop = frame[int(y1):int(y2), int(x1):int(x2)]

                    # Get face embeddings if face detected
                    embedding = self.face_detection.get_face_embedding(person_crop)
                    if embedding is not None and len(embedding) > 0:
                        self.is_new_person(embedding)

    def save_total_count_to_db(self):
        total_persons = len(self.person_database)
        record = {
            "use_case": self.use_case,
            "camera": "Webcam" if self.camera_source == 0 else "RTSP",
            "person_count": total_persons
        }
        self.mongo_handler.insert_record(record)
        print(f"Record saved to MongoDB: {record}")
