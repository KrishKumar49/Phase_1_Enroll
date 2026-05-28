import os

import psycopg2


DATABASE_URL = os.getenv("https://ghcr.io/railwayapp-templates/postgres-ssl")


def get_db_connection():
    return psycopg2.connect(DATABASE_URL)
    
    
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