import sqlite3


class SqliteAccess:
    def __init__(self):
        self.db_path = 'members.db'
        self.db_schema_sql = 'create_db.sql'

    # SQLite3-related operations
    # See SQLite3 usage pattern from Flask official doc
    # http://flask.pocoo.org/docs/0.10/patterns/sqlite3/
    @property
    def db(self):
        db = getattr(g, '_database', None)
        if db is None:
            db = g._database = sqlite3.connect(self.db_path)
            # Enable foreign key check
            db.execute("PRAGMA foreign_keys = ON")
        return db

    @staticmethod
    def close_connection():
        db = getattr(g, '_database', None)
        if db is not None:
            db.close()


