from fastapi import FastAPI, HTTPException
import os, psycopg
from psycopg.rows import dict_row

DATABASE_URL = os.getenv("DATABASE_URL")

app = FastAPI()

def get_conn():
    return psycopg.connect(DATABASE_URL, autocommit=True, row_factory=psycopg.rows.dict_row)

@app.get("/")
def get_root():
    return { "msg": "Clothing Store v0.1" }

# GET /categories 
@app.get("/categories")
def get_categories():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT category_id, name FROM categories ORDER BY category_id;")
        return cur.fetchall()

# GET /categories/{id}
@app.get("/categories/{category_id}")
def get_category(category_id: int):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT category_id, name FROM categories WHERE category_id = %s;", (category_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Category not found")
        return row

# POST /categories
@app.post("/categories", status_code=201)
def create_category(data: dict):
    name = data.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="Missing 'name'")
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO categories (name) VALUES (%s) RETURNING category_id, name;", (name,))
        return cur.fetchone()