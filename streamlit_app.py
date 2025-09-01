import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# -------------------------------
# Generate synthetic trade data
# -------------------------------
def generate_fake_trades(n=1000):
    np.random.seed(42)
    timestamps = [datetime.now() - timedelta(minutes=i) for i in range(n)]
    prices = np.random.normal(100, 2, n).round(2)  # mean 100, std 2
    volumes = np.random.randint(10, 1000, n)
    trade_ids = range(1, n+1)

    df = pd.DataFrame({
        "trade_id": trade_ids,
        "timestamp": timestamps,
        "price": prices,
        "volume": volumes
    })
    return df

# -------------------------------
# Run SQL queries on trade data
# -------------------------------
def run_sql_query(df, query):
    conn = sqlite3.connect(":memory:")
    df.to_sql("trades", conn, index=False, if_exists="replace")
    result = pd.read_sql_query(query, conn)
    conn.close()
    return result

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="TradeFlow Analyzer", layout="wide")
st.title("📊 TradeFlow Analyzer")
st.markdown("Synthetic trade data with SQL + anomaly detection")

# Generate trades
df = generate_fake_trades(5000)

# Show raw data
if st.checkbox("Show sample trades"):
    st.dataframe(df.head(20))

# Run custom SQL query
st.subheader("Run SQL on Trades")
default_query = "SELECT AVG(price) as avg_price, SUM(volume) as total_volume FROM trades"
query = st.text_area("Enter SQL query:", value=default_query, height=100)

if st.button("Run Query"):
    try:
        result = run_sql_query(df, query)
        st.dataframe(result)
    except Exception as e:
        st.error(f"SQL error: {e}")

# Liquidity patterns
st.subheader("Liquidity Patterns")
df["minute"] = df["timestamp"].dt.floor("T")
liquidity = df.groupby("minute")["volume"].sum().reset_index()

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(liquidity["minute"], liquidity["volume"], label="Total Volume per Minute")
ax.set_title("Liquidity Over Time")
ax.set_ylabel("Volume")
ax.set_xlabel("Time")
ax.legend()
st.pyplot(fig)

# Anomaly detection: Price & Volume outliers
st.subheader("Anomaly Detection")
price_mean, price_std = df["price"].mean(), df["price"].std()
outliers = df[(df["price"] > price_mean + 3*price_std) | (df["price"] < price_mean - 3*price_std)]

if not outliers.empty:
    st.warning(f"Detected {len(outliers)} abnormal trades (3σ rule)")
    st.dataframe(outliers.head(10))
else:
    st.success("No price anomalies detected.")

# Heatmap of price-volume correlation
st.subheader("Price-Volume Relationship")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="price", y="volume", alpha=0.3, ax=ax)
ax.set_title("Price vs Volume")
st.pyplot(fig)
