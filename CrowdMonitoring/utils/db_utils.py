from pymongo import MongoClient
from datetime import datetime

def get_mongo_collection(uri, db_name, collection_name):
    client = MongoClient(uri)
    db = client[db_name]
    return db[collection_name]

def insert_crowd_data(collection, screenshot_path):
    record = {
        "usecase_name": "crowd monitoring",
        "date": datetime.now().strftime("%d-%m-%Y"),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "screenshot_path": screenshot_path
    }
    collection.insert_one(record)
