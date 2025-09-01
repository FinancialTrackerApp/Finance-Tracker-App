from database import get_connection

conn = get_connection()
if conn:
    cur = conn.cursor()
    cur.execute("SELECT 'Hello from Neon!' AS message;")
    print(cur.fetchone())
    conn.close()
