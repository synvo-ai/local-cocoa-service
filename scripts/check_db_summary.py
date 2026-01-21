import sqlite3
import os

DB_PATH = r"c:\Users\tiany\Desktop\synvo-local\runtime\synvo_db\index.sqlite"

def check_summary():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, name, summary FROM files WHERE name LIKE '%2page.pdf%'")
    rows = cursor.fetchall()
    
    if not rows:
        print("No file found matching '2page.pdf'")
    else:
        for row in rows:
            file_id, name, summary = row
            print(f"File: {name} (ID: {file_id})")
            print(f"Summary length: {len(summary) if summary else 0}")
            print(f"Summary content: {summary[:200]}..." if summary else "Summary is None or empty")

    conn.close()

if __name__ == "__main__":
    check_summary()
