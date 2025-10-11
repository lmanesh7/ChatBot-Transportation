# file: tools.py

import os
from pymongo import MongoClient
from datetime import datetime
from langchain.tools import tool  # <-- Make sure this import is present

# --- It's best practice to store secrets in an environment variable ---
MONGO_URI = "mongodb+srv://lakshmanmanesh235_db_user:jsZYKWZqgJqUqa5v@cluster0.ykbqyhp.mongodb.net/"#os.environ["MONGO_URI"]
client = MongoClient(MONGO_URI)
db = client.transportationDB

# The function must be decorated with @tool
@tool
def create_work_order(description: str, location: str, priority: int) -> str:
    """
    Use this tool to create a new maintenance work order.
    Provide a description of the issue, its location, and a priority from 1 (high) to 3 (low).
    """
    print("--- Calling create_work_order tool ---")

    work_order = {
        "description": description,
        "location": location,
        "priority": priority,
        "status": "open",
        "createdAt": datetime.utcnow()
    }

    result = db.reports.insert_one(work_order)

    # Return a confirmation string to the agent
    return f"Work order created successfully. The new work order ID is {result.inserted_id}."