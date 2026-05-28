import os
from importlib import import_module


def _get_psycopg2():
    try:
        return import_module("psycopg2")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "psycopg2 is not installed. Add psycopg2-binary to requirements and redeploy."
        ) from error


def _get_connection_kwargs():
    database_url = os.getenv("DATABASE_URL")

    if database_url:
        return {"dsn": database_url}

    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "database": os.getenv("DB_NAME", "vehicle_security"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "Krish@7042"),
        "port": os.getenv("DB_PORT", "5432"),
    }

def get_db_connection():
    return _get_psycopg2().connect(**_get_connection_kwargs())
    
    
def save_employee_embedding(
    employee_id,
    mean_path,
    all_path,
    embeddings_count
):
    conn = get_db_connection()
    
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO employees (
            employee_id,
            mean_embedding_path,
            all_embedding_path,
            embeddings_count
        )
        VALUES (%s, %s, %s, %s)
        """,
        (
            employee_id,
            mean_path,
            all_path,
            embeddings_count
        )
    )
    conn.commit()
    cursor.close()
    conn.close()