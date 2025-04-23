import sqlite3
import os
import datetime
from pathlib import Path


class SecurityDatabase:
    def __init__(self, db_path='data/security_logs.db'):
        # Make sure the directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.db_path = db_path
        self.conn = None
        self.initialize_db()

    def initialize_db(self):
        """Initialize the database with the required tables."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()

        # Create the security logs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS security_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            face_id TEXT,
            is_authorized INTEGER,
            weapon_detected INTEGER,
            image_path TEXT
        )
        ''')

        self.conn.commit()
        self.conn.close()

    def add_log(self, face_id=None, is_authorized=False, weapon_detected=False, image_path=None):
        """Add a new security log entry."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute('''
        INSERT INTO security_logs (timestamp, face_id, is_authorized, weapon_detected, image_path)
        VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, face_id, int(is_authorized), int(weapon_detected), image_path))

        self.conn.commit()
        self.conn.close()

    def get_logs(self, limit=100):
        """Retrieve the latest security logs."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.cursor()

        cursor.execute('''
        SELECT * FROM security_logs ORDER BY timestamp DESC LIMIT ?
        ''', (limit,))

        logs = [dict(row) for row in cursor.fetchall()]

        self.conn.close()

        return logs

    def get_logs_by_date(self, start_date, end_date):
        """Retrieve logs between specific dates."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.cursor()

        cursor.execute('''
        SELECT * FROM security_logs 
        WHERE timestamp BETWEEN ? AND ?
        ORDER BY timestamp DESC
        ''', (start_date, end_date))

        logs = [dict(row) for row in cursor.fetchall()]

        self.conn.close()

        return logs