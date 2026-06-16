import os

import psycopg2
from dotenv import load_dotenv


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def get_db_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")

    return psycopg2.connect(DATABASE_URL)


def save_employee_embedding(
    employee_id,
    mean_embedding,
    all_embeddings,
    embeddings_count
):
    conn = get_db_connection()

    cursor = conn.cursor()

    mean_embedding_list = mean_embedding.tolist()

    cursor.execute(
        """
        INSERT INTO employees (
            employee_id,
            mean_embedding,
            embeddings_count
        )
        VALUES (%s, %s, %s)
        ON CONFLICT (employee_id)
        DO UPDATE SET
            mean_embedding = EXCLUDED.mean_embedding,
            embeddings_count = EXCLUDED.embeddings_count
        """,
        (
            employee_id,
            mean_embedding_list,
            embeddings_count
        )
    )

    cursor.execute(
    """
    DELETE FROM employee_embeddings
    WHERE employee_id = %s
    """,
    (employee_id,)
    )
    
    for embedding in all_embeddings:
        embedding_list = embedding.tolist()

        cursor.execute(
            """
            INSERT INTO employee_embeddings (
                employee_id,
                embedding
            )
            VALUES (%s, %s)
            """,
            (
                employee_id,
                embedding_list
            )
        )

    conn.commit()

    cursor.close()
    conn.close()
    
    
def get_all_employee_embeddings():
    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            employee_id,
            embedding
        FROM employee_embeddings
        """
    )

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    return rows
