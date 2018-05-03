from pymongo import MongoClient
import pprint

client = MongoClient('mongodb://localhost:27017/')
db = client['local']
collection = db['startup_log']

print(db.collection_names(include_system_collections=False))
# pprint.pprint(db.startup_log.find_one())


def insert_test(cnt):
    test = {"id": 1,
            "cnt": cnt}
    collection.insert_one(test)

