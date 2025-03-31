# from pymongo import MongoClient
# import datetime

# class Database:
#     """Class to handle database operations"""
    
#     def __init__(self, host, port, db_name, collection_name):
#         """Initialize the database connection
        
#         Args:
#             host (str): MongoDB host
#             port (int): MongoDB port
#             db_name (str): Database name
#             collection_name (str): Collection name
#         """
#         self.client = MongoClient(host, port)
#         self.db = self.client[db_name]
#         self.collection = self.db[collection_name]
    
#     def save_accident_record(self, screenshot_path):
#         """Save accident information to MongoDB
        
#         Args:
#             screenshot_path (str): Path to the accident screenshot
            
#         Returns:
#             str: ID of the inserted record
#         """
#         current_time = datetime.datetime.now()
#         time_str = current_time.strftime("%H:%M:%S")
#         date_str = current_time.strftime("%d-%m-%Y")
        
#         record = {
#             "port_accident": {
#                 "accidents": [
#                     {
#                         "use_case": "Port Accident Detection",
#                         "timestamp": time_str,
#                         "date": date_str,
#                         "screenshot": screenshot_path
#                     }
#                 ]
#             }
#         }
        
#         result = self.collection.insert_one(record)
#         print(f"Accident record saved to MongoDB database: {record}")
#         return str(result.inserted_id)



from pymongo import MongoClient
import datetime

class Database:
    """Class to handle database operations"""

    def __init__(self, host, port, db_name, collection_name):
        """Initialize the database connection
        
        Args:
            host (str): MongoDB host
            port (int): MongoDB port
            db_name (str): Database name
            collection_name (str): Collection name
        """
        self.client = MongoClient(host, port)
        self.db_name = db_name
        self.collection_name = collection_name
        
        # Dynamically create the database and collection if they do not exist
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]

    def save_accident_record(self, screenshot_path):
        """Save accident information to MongoDB
        
        Args:
            screenshot_path (str): Path to the accident screenshot
            
        Returns:
            str: ID of the inserted record
        """
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%H:%M:%S")
        date_str = current_time.strftime("%d-%m-%Y")

        # Structure of the accident record
        record = {
            "port_accident": {
                "accidents": [
                    {
                        "use_case": "Port Accident Detection",
                        "timestamp": time_str,
                        "date": date_str,
                        "screenshot": screenshot_path
                    }
                ]
            }
        }

        # Insert the record into the collection
        result = self.collection.insert_one(record)
        print(f"Accident record saved to MongoDB database: {record}")
        return str(result.inserted_id)

# Example usage
if __name__ == "__main__":
    db = Database(host="localhost", port=27017, db_name="AccidentDB", collection_name="Accidents")
    screenshot_path = "/path/to/screenshot.png"
    record_id = db.save_accident_record(screenshot_path)
    print(f"Inserted record ID: {record_id}")
