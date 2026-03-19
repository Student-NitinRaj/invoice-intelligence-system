import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


# ------------------ LOAD DATA ------------------ #
def load_invoice_data():
    db_path = Path(__file__).resolve().parent.parent / "Data" / "inventory.db"

    conn = sqlite3.connect(db_path)

    query = """
    WITH purchase_agg AS (
        SELECT
            p.PONumber,
            COUNT(DISTINCT p.Brand) AS total_brands,
            SUM(p.Quantity) AS total_item_quantity,
            SUM(p.Dollars) AS total_item_dollars,
            AVG(julianday(p.ReceivingDate) - julianday(p.PODate)) AS avg_receiving_delay
        FROM purchases p
        GROUP BY p.PONumber
    )
    SELECT
        vi.PONumber,
        vi.Quantity AS invoice_quantity,
        vi.Dollars AS invoice_dollars,
        vi.Freight,
        (julianday(vi.InvoiceDate) - julianday(vi.PODate)) AS days_po_to_invoice,
        (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) AS days_to_pay,
        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay
    FROM vendor_invoice vi
    LEFT JOIN purchase_agg pa
        ON vi.PONumber = pa.PONumber
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


# ------------------ CREATE LABEL ------------------ #
def create_invoice_risk_label(row):
    if abs(row["invoice_dollars"] - row["total_item_dollars"]) > 5:
        return 1
    if row["avg_receiving_delay"] > 10:
        return 1
    return 0


# ------------------ APPLY LABEL ------------------ #
def apply_labels(df):
    df["flag_invoice"] = df.apply(create_invoice_risk_label, axis=1)
    return df


# ------------------ SPLIT DATA ------------------ #
def split_data(df, features, target):
    df = df.dropna()  # ⚠️ important (avoid model crash)

    x = df[features]
    y = df[target]

    return train_test_split(
        x, y, test_size=0.2, random_state=42
    )


# ------------------ SCALE FEATURES ------------------ #
def scale_features(x_train, x_test, scaler_path):
    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    joblib.dump(scaler, scaler_path)

    return x_train_scaled, x_test_scaled