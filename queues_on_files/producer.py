import csv
import uuid
from datetime import datetime

QUEUE_FILE = "queue.csv"
TASK_COUNT = 100

def add_tasks():
    with open(QUEUE_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)

        for _ in range(TASK_COUNT):
            task_id = str(uuid.uuid4())
            created_at = datetime.now().isoformat()
            writer.writerow([task_id, "pending", created_at])
            print(f"Dodano zadanie {task_id}")

if __name__ == "__main__":
    add_tasks()