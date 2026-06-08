import os

import psycopg2
from dotenv import load_dotenv


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def get_db_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")

    return psycopg2.connect(DATABASE_URL)


def _to_pgvector_literal(values):
    if values is None:
        return None

    if hasattr(values, "tolist"):
        values = values.tolist()

    return "[" + ",".join(str(float(value)) for value in values) + "]"


def get_active_visit_id(employee_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT visit_id
        FROM employee_visits
        WHERE employee_id = %s
          AND status = 'ACTIVE'
        ORDER BY entry_time DESC
        LIMIT 1
        """,
        (employee_id,)
    )

    row = cursor.fetchone()

    cursor.close()
    conn.close()

    if row is None:
        return None

    return row[0]


def save_vehicle_entry_record(
    employee_id,
    visit_id,
    vehicle_embedding,
    vehicle_class,
    plate_number=None,
    camera_id=None,
):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO entry_records (
            visit_id,
            employee_id,
            vehicle_embedding,
            vehicle_class,
            plate_number,
            camera_id
        )
        VALUES (%s, %s, %s::vector, %s, %s, %s)
        RETURNING id
        """,
        (
            visit_id,
            employee_id,
            _to_pgvector_literal(vehicle_embedding),
            vehicle_class,
            plate_number,
            camera_id,
        )
    )

    entry_id = cursor.fetchone()[0]

    conn.commit()
    cursor.close()
    conn.close()

    return entry_id


def save_employee_embedding(
    employee_id,
    mean_embedding,
    all_embeddings,
    embeddings_count
):
    conn = get_db_connection()

    cursor = conn.cursor()

    mean_embedding_list = _to_pgvector_literal(mean_embedding)

    cursor.execute(
        """
        INSERT INTO employees (
            employee_id,
            mean_embedding,
            embeddings_count
        )
        VALUES (%s, %s::vector, %s)
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