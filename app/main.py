from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import bcrypt
import psycopg
from psycopg.rows import dict_row
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from jose import jwt, JWTError
from dotenv import load_dotenv

# -----------------------------
# Load .env reliably (fixes your 500 DATABASE_URL missing)
# app/main.py -> project root is two levels up
# -----------------------------
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)

DATABASE_URL = os.getenv("DATABASE_URL")
JWT_SECRET = os.getenv("JWT_SECRET", "change_me_to_a_long_random_string")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

ALGORITHM = "HS256"

app = FastAPI(title="Clothing Store API with Analytics", version="0.1.0")

bearer = HTTPBearer(auto_error=False)


# -----------------------------
# DB helpers
# -----------------------------
def require_database_url() -> str:
    if not DATABASE_URL:
        # Return exactly the kind of message you saw in Swagger
        raise HTTPException(
            status_code=500,
            detail="DATABASE_URL is missing. Put it in .env and run uvicorn with --env-file .env",
        )
    return DATABASE_URL


def get_conn() -> psycopg.Connection:
    return psycopg.connect(require_database_url(), row_factory=dict_row)


# -----------------------------
# Security helpers (bcrypt + JWT)
# -----------------------------
def hash_password(plain: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(plain.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def create_access_token(subject: str, role: str, email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": subject,     # customer_id as string
        "role": role,
        "email": email,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)


def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
) -> Dict[str, Any]:
    if creds is None or not creds.credentials:
        raise HTTPException(status_code=401, detail="Login required (Bearer token missing).")

    token = creds.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        sub = payload.get("sub")
        role = payload.get("role", "customer")
        email = payload.get("email")

        if not sub:
            raise HTTPException(status_code=401, detail="Invalid token (missing subject).")

        return {"customer_id": int(sub), "role": role, "email": email}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


def require_admin(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user


# -----------------------------
# Pydantic models
# -----------------------------
class CategoryCreate(BaseModel):
    name: str = Field(..., min_length=1)


class CategoryOut(BaseModel):
    category_id: int
    name: str


class ProductCreate(BaseModel):
    name: str = Field(..., min_length=1)
    category_id: int
    price: float = Field(..., ge=0)
    stock: int = Field(..., ge=0)


class ProductPatch(BaseModel):
    name: Optional[str] = Field(None, min_length=1)
    category_id: Optional[int] = None
    price: Optional[float] = Field(None, ge=0)
    stock: Optional[int] = Field(None, ge=0)


class ProductOut(BaseModel):
    product_id: int
    name: str
    category: str
    price: float
    stock: int


class OrderCreate(BaseModel):
    user_id: int
    product_id: int
    quantity: int = Field(..., gt=0)


class OrderItemOut(BaseModel):
    product_id: int
    name: str
    price: float
    quantity: int
    line_total: float


class OrderOut(BaseModel):
    order_id: int
    customer_id: int
    order_date: str
    items: List[OrderItemOut]
    order_total: float


class UserRegister(BaseModel):
    # Swagger shows all these fields; we'll accept them safely
    name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: str
    password: str


class UserOut(BaseModel):
    customer_id: int
    first_name: str
    last_name: str
    email: str
    role: str


class UserLogin(BaseModel):
    email: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


# -----------------------------
# Basic
# -----------------------------
@app.get("/")
def get_root():
    return {"msg": "Clothing Store v0.1"}


# -----------------------------
# Categories (GET open, POST admin)
# -----------------------------
@app.get("/categories", response_model=List[CategoryOut])
def list_categories():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT category_id, name FROM categories ORDER BY category_id;")
        return cur.fetchall()


@app.get("/categories/{category_id}", response_model=CategoryOut)
def get_category(category_id: int):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT category_id, name FROM categories WHERE category_id = %s;",
            (category_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Category not found")
        return row


@app.post("/categories", status_code=201, response_model=CategoryOut)
def create_category(data: CategoryCreate, _: Dict[str, Any] = Depends(require_admin)):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO categories (name) VALUES (%s) RETURNING category_id, name;",
            (data.name,),
        )
        return cur.fetchone()


# -----------------------------
# Required: GET /products
# product name, category name, price, stock
# -----------------------------
@app.get("/products", response_model=List[ProductOut])
def list_products():
    sql = """
    SELECT
      p.product_id,
      p.name,
      c.name AS category,
      p.price,
      p.stock
    FROM products p
    JOIN categories c ON c.category_id = p.category_id
    ORDER BY p.product_id;
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


# Admin CRUD (optional / extra)
@app.post("/products", status_code=201)
def create_product(data: ProductCreate, _: Dict[str, Any] = Depends(require_admin)):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO products (name, category_id, price, stock)
            VALUES (%s, %s, %s, %s)
            RETURNING product_id, name, category_id, price, stock;
            """,
            (data.name, data.category_id, data.price, data.stock),
        )
        return cur.fetchone()


@app.patch("/products/{product_id}")
def patch_product(product_id: int, data: ProductPatch, _: Dict[str, Any] = Depends(require_admin)):
    fields = []
    values = []

    if data.name is not None:
        fields.append("name = %s")
        values.append(data.name)
    if data.category_id is not None:
        fields.append("category_id = %s")
        values.append(data.category_id)
    if data.price is not None:
        fields.append("price = %s")
        values.append(data.price)
    if data.stock is not None:
        fields.append("stock = %s")
        values.append(data.stock)

    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update.")

    values.append(product_id)

    sql = f"""
    UPDATE products
    SET {", ".join(fields)}
    WHERE product_id = %s
    RETURNING product_id, name, category_id, price, stock;
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, tuple(values))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Product not found")
        return row


@app.delete("/products/{product_id}", status_code=204)
def delete_product(product_id: int, _: Dict[str, Any] = Depends(require_admin)):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM products WHERE product_id = %s;", (product_id,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Product not found")
    return None


# -----------------------------
# OPTIONAL AUTH: Register + Login
# -----------------------------
@app.post("/users", response_model=UserOut, status_code=201)
def register_user(payload: UserRegister):
    # Build first/last name from payload
    first_name = (payload.first_name or "").strip()
    last_name = (payload.last_name or "").strip()

    if (not first_name or not last_name) and payload.name:
        parts = payload.name.strip().split()
        if len(parts) == 1:
            first_name = parts[0]
            last_name = ""
        else:
            first_name = parts[0]
            last_name = " ".join(parts[1:])

    if not first_name:
        raise HTTPException(status_code=400, detail="first_name (or name) is required.")
    if not payload.email:
        raise HTTPException(status_code=400, detail="email is required.")
    if not payload.password:
        raise HTTPException(status_code=400, detail="password is required.")

    pw_hash = hash_password(payload.password)

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Ensure optional columns exist (otherwise you get 500)
            # If columns already exist, this does nothing.
            cur.execute("ALTER TABLE customers ADD COLUMN IF NOT EXISTS password VARCHAR(255);")
            cur.execute("ALTER TABLE customers ADD COLUMN IF NOT EXISTS role VARCHAR(50) DEFAULT 'customer';")

            # Email uniqueness check
            cur.execute("SELECT customer_id FROM customers WHERE email = %s;", (payload.email,))
            if cur.fetchone():
                raise HTTPException(status_code=409, detail="Email already registered.")

            cur.execute(
                """
                INSERT INTO customers (first_name, last_name, email, password, role)
                VALUES (%s, %s, %s, %s, 'customer')
                RETURNING customer_id, first_name, last_name, email, role;
                """,
                (first_name, last_name, payload.email, pw_hash),
            )
            row = cur.fetchone()
        conn.commit()
        return row


@app.post("/users/login", response_model=TokenOut)
def login_user(payload: UserLogin):
    with get_conn() as conn, conn.cursor() as cur:
        # Expect columns password + role exist when auth is enabled
        cur.execute(
            """
            SELECT customer_id, email, password, role
            FROM customers
            WHERE email = %s;
            """,
            (payload.email,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        if not row.get("password") or not verify_password(payload.password, row["password"]):
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        token = create_access_token(
            subject=str(row["customer_id"]),
            role=row.get("role", "customer") or "customer",
            email=row["email"],
        )
        return {"access_token": token, "token_type": "bearer"}


@app.delete("/users/{customer_id}", status_code=204)
def delete_user(customer_id: int, _: Dict[str, Any] = Depends(require_admin)):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM customers WHERE customer_id = %s;", (customer_id,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")
    return None


# -----------------------------
# Orders (auth required)
# Logged-in customers: can place orders & view own
# Admin: can view all with ?all=true and can place for any user_id
# -----------------------------
@app.post("/orders", response_model=OrderOut, status_code=201)
def create_order(payload: OrderCreate, user: Dict[str, Any] = Depends(get_current_user)):
    # If not admin, force user_id to match token user
    if user["role"] != "admin" and payload.user_id != user["customer_id"]:
        raise HTTPException(status_code=403, detail="You can only place orders for yourself.")

    with get_conn() as conn:
        try:
            with conn.cursor() as cur:
                # Ensure customer exists
                cur.execute(
                    "SELECT customer_id FROM customers WHERE customer_id = %s;",
                    (payload.user_id,),
                )
                if not cur.fetchone():
                    raise HTTPException(status_code=404, detail="Customer not found")

                # Lock product row to safely decrease stock
                cur.execute(
                    """
                    SELECT product_id, name, price, stock
                    FROM products
                    WHERE product_id = %s
                    FOR UPDATE;
                    """,
                    (payload.product_id,),
                )
                product = cur.fetchone()
                if not product:
                    raise HTTPException(status_code=404, detail="Product not found")

                if int(product["stock"]) < payload.quantity:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Not enough stock. Available={product['stock']}, requested={payload.quantity}",
                    )

                # Create order
                cur.execute(
                    """
                    INSERT INTO orders (customer_id)
                    VALUES (%s)
                    RETURNING order_id, customer_id, order_date;
                    """,
                    (payload.user_id,),
                )
                order = cur.fetchone()

                # Insert order item (store current price as unit_price)
                cur.execute(
                    """
                    INSERT INTO order_items (order_id, product_id, quantity, unit_price)
                    VALUES (%s, %s, %s, %s)
                    RETURNING product_id, quantity, unit_price;
                    """,
                    (order["order_id"], product["product_id"], payload.quantity, product["price"]),
                )
                item = cur.fetchone()

                # Decrease stock
                cur.execute(
                    "UPDATE products SET stock = stock - %s WHERE product_id = %s;",
                    (payload.quantity, payload.product_id),
                )

            conn.commit()

            line_total = float(item["quantity"]) * float(item["unit_price"])
            resp = {
                "order_id": order["order_id"],
                "customer_id": order["customer_id"],
                "order_date": str(order["order_date"]),
                "items": [
                    {
                        "product_id": product["product_id"],
                        "name": product["name"],
                        "price": float(item["unit_price"]),
                        "quantity": int(item["quantity"]),
                        "line_total": line_total,
                    }
                ],
                "order_total": line_total,
            }
            return resp

        except HTTPException:
            conn.rollback()
            raise
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/orders", response_model=List[OrderOut])
def list_orders(
    all: bool = Query(False, description="Admin only: list all orders if true"),
    user: Dict[str, Any] = Depends(get_current_user),
):
    if all and user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required to list all orders.")

    with get_conn() as conn, conn.cursor() as cur:
        if all and user["role"] == "admin":
            cur.execute(
                "SELECT order_id, customer_id, order_date FROM orders ORDER BY order_id DESC;"
            )
        else:
            cur.execute(
                """
                SELECT order_id, customer_id, order_date
                FROM orders
                WHERE customer_id = %s
                ORDER BY order_id DESC;
                """,
                (user["customer_id"],),
            )
        orders = cur.fetchall()

        results: List[Dict[str, Any]] = []
        for o in orders:
            cur.execute(
                """
                SELECT
                  oi.product_id,
                  p.name,
                  oi.unit_price AS price,
                  oi.quantity,
                  (oi.unit_price * oi.quantity) AS line_total
                FROM order_items oi
                JOIN products p ON p.product_id = oi.product_id
                WHERE oi.order_id = %s
                ORDER BY oi.product_id;
                """,
                (o["order_id"],),
            )
            items = cur.fetchall()
            order_total = float(sum(float(i["line_total"]) for i in items)) if items else 0.0

            results.append(
                {
                    "order_id": o["order_id"],
                    "customer_id": o["customer_id"],
                    "order_date": str(o["order_date"]),
                    "items": items,
                    "order_total": order_total,
                }
            )

        return results


# -----------------------------
# Required: statistics (SQL aggregates + joins)
# Protected as ADMIN in optional auth mode
# -----------------------------
@app.get("/statistics/users")
def stats_users(_: Dict[str, Any] = Depends(require_admin)):
    sql = """
    SELECT
      c.customer_id AS user_id,
      c.first_name,
      c.last_name,
      COUNT(DISTINCT o.order_id) AS order_count,
      COALESCE(SUM(oi.quantity), 0) AS items_bought,
      COALESCE(SUM(oi.quantity * oi.unit_price), 0) AS money_spent
    FROM customers c
    LEFT JOIN orders o ON o.customer_id = c.customer_id
    LEFT JOIN order_items oi ON oi.order_id = o.order_id
    GROUP BY c.customer_id, c.first_name, c.last_name
    ORDER BY money_spent DESC, order_count DESC;
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


@app.get("/statistics/products")
def stats_products(_: Dict[str, Any] = Depends(require_admin)):
    sql = """
    SELECT
      p.product_id,
      p.name,
      COUNT(DISTINCT oi.order_id) AS order_count,
      COALESCE(SUM(oi.quantity), 0) AS units_sold,
      COALESCE(SUM(oi.quantity * oi.unit_price), 0) AS turnover
    FROM products p
    LEFT JOIN order_items oi ON oi.product_id = p.product_id
    GROUP BY p.product_id, p.name
    ORDER BY turnover DESC, units_sold DESC;
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()
