# python
"""
Простой скрипт для подключения к Postgres и создания тестовой таблицы.
Установка зависимости (Windows):
    pip install psycopg2-binary
Переменные окружения (при необходимости):
    PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
"""
import os
import sys
import psycopg2
from psycopg2 import sql

def get_conn_params():
    return {
        "host": os.environ.get("PGHOST", "localhost"),
        "port": int(os.environ.get("PGPORT", 5432)),
        "dbname": os.environ.get("PGDATABASE", "test_db"),
        "user": os.environ.get("PGUSER", "postgres"),
        "password": os.environ.get("PGPASSWORD", "123456"),
    }

def create_test_table(conn):
    create_query = """
    CREATE TABLE IF NOT EXISTS test_table_2 (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
    );
    """
    with conn.cursor() as cur:
        cur.execute(create_query)
    conn.commit()

def main():
    params = get_conn_params()
    try:
        conn = psycopg2.connect(**params)
    except Exception as e:
        print(f"Connection error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        create_test_table(conn)
        print("Таблица 'test_table' успешно создана или уже существует.")
    except Exception as e:
        print(f"Ошибка при создании таблицы: {e}", file=sys.stderr)
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    main()
