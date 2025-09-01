import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

st.set_page_config(page_title="Trade Data Analysis", layout="wide")

st.title("📊 Trade Data Analysis Dashboard")
st.markdown("SQL + Python pipeline for liquidity and anomaly detection.")

# File uploader
uploaded_file = st.file_uploader("Upload trade data (CSV)", type="csv")

if uploaded_file:
    # Load CSV into DataFrame
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # Basic checks
    required_cols = {"trade_id", "timestamp", "price", "volume"}
    if not required_cols.issubset(df.columns):
        st.error(f"CSV must contain: {required_cols}")
    else:
        # Load into SQLite
        conn = sqlite3.connect(":memory:")
        df.to_sql("trades", conn, index=False, if_exists="replace")

        st.subheader("Liquidity Patterns (SQL Query)")
        query = """
        SELECT 
            strftime('%Y-%m-%d %H:00:00', timestamp) AS trade_hour,
            SUM(volume) as total_volume,
            AVG(price) as avg_price
        FROM trades
        GROUP BY trade_hour
        ORDER BY trade_hour;
        """
        result = pd.read_sql(query, conn)
        st.dataframe(result.head())

        # Plot liquidity over time
        fig, ax = plt.subplots()
        ax.plot(result["trade_hour"], result["total_volume"], marker="o")
        plt.xticks(rotation=45)
        ax.set_title("Liquidity (Total Volume per Hour)")
        ax.set_ylabel("Volume")
        st.pyplot(fig)

        # --- Anomaly detection ---
        st.subheader("Anomaly Detection (Python)")
        df["z_score_volume"] = (df["volume"] - df["volume"].mean()) / df["volume"].std()
        anomalies = df[df["z_score_volume"].abs() > 3]

        st.write(f"Detected {len(anomalies)} anomalous trades (|z| > 3):")
        st.dataframe(anomalies)

        # Plot anomalies
        fig2, ax2 = plt.subplots()
        ax2.scatter(df["timestamp"], df["volume"], alpha=0.5, label="Normal")
        ax2.scatter(anomalies["timestamp"], anomalies["volume"], color="red", label="Anomaly")
        plt.xticks(rotation=45)
        ax2.set_title("Trade Volumes with Anomalies")
        ax2.legend()
        st.pyplot(fig2)
