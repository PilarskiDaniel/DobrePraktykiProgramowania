import csv
import time
from datetime import datetime

QUEUE_FILE = "queue.csv"
CHECK_INTERVAL = 5    # co ile sekund sprawdzamy kolejkę
TASK_DURATION = 30    # czas wykonania zadania (30s)

def read_tasks():
    try:
        with open(QUEUE_FILE, mode="r", newline="") as file:
            return list(csv.reader(file))
    except FileNotFoundError:
        return []

def write_tasks(tasks):
    with open(QUEUE_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(tasks)

def consume_task():
    tasks = read_tasks()

    for task in tasks:
        task_id, status, created_at = task

        if status == "pending":
            print(f"[{datetime.now()}] Pobieram zadanie {task_id}")

            # zmiana statusu na in_progress
            task[1] = "in_progress"
            write_tasks(tasks)

            print(f"[{datetime.now()}] Wykonuję zadanie {task_id}")
            time.sleep(TASK_DURATION)

            # zmiana statusu na done
            task[1] = "done"
            write_tasks(tasks)

            print(f"[{datetime.now()}] Zakończono zadanie {task_id}")
            return

if __name__ == "__main__":
    print("Consumer uruchomiony...")
    while True:
        consume_task()
        time.sleep(CHECK_INTERVAL)