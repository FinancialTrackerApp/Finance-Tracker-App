import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get the DB URL
DATABASE_URL = os.getenv("DATABASE_URL")

def get_connection():
    try:
        conn = psycopg2.connect(DATABASE_URL, sslmode="require", cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        print("❌ Error connecting to Neon:", e)
        return None
