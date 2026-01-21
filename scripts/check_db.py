import sqlite3
import os

db_path = "../runtime/synvo_db/index.sqlite"

if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

try:
    cursor.execute("SELECT id, label, protocol, username, client_id, tenant_id FROM email_accounts")
    rows = cursor.fetchall()
    with open("db_check_output.txt", "w") as f:
        f.write(f"Found {len(rows)} accounts:\n")
        for row in rows:
            f.write(f"ID: {row['id']}, Label: {row['label']}, Protocol: {row['protocol']}, Username: {row['username']}, ClientID: {row['client_id']}, TenantID: {row['tenant_id']}\n")
    print("Output written to db_check_output.txt")
except Exception as e:
    with open("db_check_output.txt", "w") as f:
        f.write(f"Error querying database: {e}\n")
    print(f"Error querying database: {e}")
finally:
    conn.close()
