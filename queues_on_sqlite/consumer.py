import sqlite3
import time
from datetime import datetime

DB_FILE = "queue.db"
CHECK_INTERVAL = 5
TASK_DURATION = 30

def consume_task():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # pobranie jednego zadania pending
    cursor.execute("""
        SELECT id FROM tasks
        WHERE status = 'pending'
        ORDER BY created_at
        LIMIT 1
    """)

    row = cursor.fetchone()

    if row is None:
        conn.close()
        return

    task_id = row[0]

    print(f"[{datetime.now()}] Pobieram zadanie {task_id}")

    # oznaczenie jako in_progress
    cursor.execute("""
        UPDATE tasks
        SET status = 'in_progress'
        WHERE id = ?
    """, (task_id,))
    conn.commit()

    print(f"[{datetime.now()}] Wykonuję zadanie {task_id}")
    time.sleep(TASK_DURATION)

    # oznaczenie jako done
    cursor.execute("""
        UPDATE tasks
        SET status = 'done'
        WHERE id = ?
    """, (task_id,))
    conn.commit()

    print(f"[{datetime.now()}] Zakończono zadanie {task_id}")

    conn.close()

if __name__ == "__main__":
    print("Consumer uruchomiony...")
    while True:
        consume_task()
        time.sleep(CHECK_INTERVAL)