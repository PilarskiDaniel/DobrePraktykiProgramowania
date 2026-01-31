import sqlite3
import uuid
from datetime import datetime

DB_FILE = "queue.db"
TASK_COUNT = 100

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

def add_tasks():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for _ in range(TASK_COUNT):
        task_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()

        cursor.execute(
            "INSERT INTO tasks (id, status, created_at) VALUES (?, ?, ?)",
            (task_id, "pending", created_at)
        )

        print(f"Dodano zadanie {task_id}")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    add_tasks()