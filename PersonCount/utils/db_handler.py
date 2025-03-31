import pymongo
from datetime import datetime

class MongoDBHandler:
    def __init__(self, db_name, collection_name, host='localhost', port=27017, camera_name='Camera_1', video_source=''):
        # Establish MongoDB connection
        try:
            self.client = pymongo.MongoClient(host, port, serverSelectionTimeoutMS=5000)
            self.client.server_info()  # Trigger exception if connection fails
            print("Connected to MongoDB successfully.")
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")
            raise 

        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.camera_name = camera_name
        self.video_source = video_source

    def insert_person_event(self, total_persons):
        """
        Insert a person detection event into MongoDB
        
        :param total_persons: Number of persons detected
        """
        event = {
            "event_id": f"id_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "video_source": self.video_source,
            "use_case": "person_count",
            "camera_name": self.camera_name,
            "screenshot": None,
            "event_data": [
                {
                    "output_name": "total_person",
                    "value": str(total_persons)
                }
            ]
        }
        
        return self.collection.insert_one(event)
