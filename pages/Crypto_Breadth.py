import streamlit as st
import psycopg
import pandas as pd
import plotly.graph_objects as go

DB_URL = st.secrets["DB_URL"]
st.set_page_config("Crypto Breadth", layout="wide")
st.warning("🚧 Dashboard is under construction — new tools will appear soon.")
###############################

@st.cache_data(ttl=3600)
def load_breadth():
    """Load the breadth table into a pandas DataFrame."""
    sql = """
        SELECT
            date,
            sma30,
            sma60,
            sma90
        FROM breadth
        ORDER BY date ASC;
    """

    with psycopg.connect(DB_URL) as conn:
        df = pd.read_sql(sql, conn)

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    return df

@st.cache_data(ttl=3600)
def load_prices():
    with psycopg.connect(DB_URL) as conn:
        df = pd.read_sql("SELECT * FROM prices ORDER BY date", conn)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


###############################
df = load_breadth()
df_btc = load_prices()

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index, y=df['sma90'], mode="lines", name="90 Day Crypto Breadth"
))
fig.add_trace(go.Scatter(
    x=df.index, y=df['sma60'], mode="lines", name="60 Day Crypto Breadth"
))
fig.add_trace(go.Scatter(
    x=df.index, y=df['sma30'], mode="lines", name="30 Day Crypto Breadth"
))

fig.add_trace(go.Scatter(
    x=df_btc.index, y=df_btc['btc'], mode="lines", name="BTC Price", yaxis="y2"
))

fig.update_layout(
    title="Crypto Breadth",
    xaxis_title="Date",
    yaxis_title="Breadth Index",
    template="plotly_dark",
    height=800,

    # Rangeslider
    xaxis=dict(
        rangeslider=dict(visible=True, thickness=0.1),
        type="date"
    ),

    # Axes
    yaxis=dict(title="Breadth Index"),
    yaxis2=dict(
        title="BTC Price (USD)",
        overlaying="y",
        side="right",
        type="log",
        showgrid=False
    )
)

st.plotly_chart(fig, use_container_width=True)
