import streamlit as st
import psycopg
import pandas as pd
import plotly.graph_objects as go

DB_URL = st.secrets["DB_URL"]

@st.cache_data(ttl=3600)
def load_market_health():
    """Load updated market health table from DB."""
    sql = """
        SELECT
            date,
            pct_higher_high,
            pct_positive_30d,
            outperform_30d,
            outperform_60d,
            outperform_90d
        FROM market_health
        ORDER BY date ASC;
    """

    with psycopg.connect(DB_URL) as conn:
        df = pd.read_sql(sql, conn)

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


@st.cache_data(ttl=3600)
def load_prices():
    """Load BTC prices for overlay."""
    with psycopg.connect(DB_URL) as conn:
        df = pd.read_sql("SELECT * FROM prices ORDER BY date", conn)

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


st.set_page_config("Market Health Dashboard", layout="wide")

df = load_market_health()
df_btc = load_prices()

st.title("Crypto Market Health Dashboard")

st.subheader("Market Structure — Higher Highs & Positive 30D Returns")
st.text("""
   • pct_higher_high — % of coins making short-term higher highs  
   • pct_positive_30d — % of coins with positive 30-day returns  
""")
fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=df.index, y=df["pct_higher_high"],
    mode="lines", name="% Higher Highs"
))
fig1.add_trace(go.Scatter(
    x=df.index, y=df["pct_positive_30d"],
    mode="lines", name="% Positive 30D Returns"
))

# BTC overlay
fig1.add_trace(go.Scatter(
    x=df_btc.index, y=df_btc["btc"],
    mode="lines", name="BTC Price (log)",
    yaxis="y2", line=dict(width=1.5)
))

fig1.update_layout(
    template="plotly_dark",
    height=500,
    xaxis=dict(
        rangeslider=dict(visible=True, thickness=0.1),
        type="date"
    ),
    yaxis=dict(title="Percentage (%)"),
    yaxis2=dict(
        title="BTC Price (USD)",
        overlaying="y", side="right", type="log", showgrid=False
    )
)

st.plotly_chart(fig1, use_container_width=True)


st.subheader("Altcoin Season Index — Outperformance vs BTC")
st.text("""        
   • outperform_30d — % of altcoins beating BTC over 30 days  
   • outperform_60d — % of altcoins beating BTC over 60 days  
   • outperform_90d — % of altcoins beating BTC over 90 days  
""")
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=df.index, y=df["outperform_30d"],
    mode="lines", name="Outperform 30D (%)"
))
fig2.add_trace(go.Scatter(
    x=df.index, y=df["outperform_60d"],
    mode="lines", name="Outperform 60D (%)"
))
fig2.add_trace(go.Scatter(
    x=df.index, y=df["outperform_90d"],
    mode="lines", name="Outperform 90D (%)"
))

# BTC overlay
fig2.add_trace(go.Scatter(
    x=df_btc.index, y=df_btc["btc"],
    mode="lines", name="BTC Price (log)",
    yaxis="y2", line=dict(width=1.5)
))

fig2.update_layout(
    template="plotly_dark",
    height=550,
    xaxis=dict(
        rangeslider=dict(visible=True, thickness=0.1),
        type="date"
    ),
    yaxis=dict(title="Altcoin Outperformance (%)"),
    yaxis2=dict(
        title="BTC Price (USD)",
        overlaying="y", side="right", type="log", showgrid=False
    )
)

st.plotly_chart(fig2, use_container_width=True)


st.markdown("---")
with st.expander("Full Market Health Data"):
    st.dataframe(df)

with st.expander("Source Code"):
    st.code(
'''
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import psycopg
import tqdm

def fetch_ohlcv_data(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str = '1d',
    exchange_name: str = 'binance'
) -> pd.DataFrame:

    # Initialize exchange
    try:
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class()
    except AttributeError:
        raise ValueError(f"Exchange '{exchange_name}' is not supported by CCXT.")

    # Convert dates to timestamps
    since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

    ohlcv = []

    # Paginated fetch
    while since < end_timestamp:
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not data:
                break

            ohlcv.extend(data)
            since = data[-1][0] + 1

        except Exception as e:
            print(f"Error fetching data for {symbol} on {exchange_name}: {e}")
            break

    if not ohlcv:
        raise ValueError(f"No data returned for {symbol} on {exchange_name}.")

    df = pd.DataFrame(
        ohlcv,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('time', inplace=True)

    return df[['open', 'high', 'low', 'close', 'volume']]

def get_usdt_pairs(exchange_name="binance"):
    exchange = getattr(ccxt, exchange_name)()
    markets = exchange.load_markets()
    return [s for s in markets.keys() if s.endswith("/USDT")]


def get_symbols():
    symbols = [
        "BTC","ETH","XRP","BNB","SOL","TRX","DOGE","ADA","ZEC","HYPE","BCH","LINK",
        "LEO","XLM","LTC","XMR","HBAR","AVAX","SUI","SHIB","UNI","TON","DOT","CRO",
        "WLFI","CC","M","MNT","TAO","NEAR","ICP","ASTER","AAVE","BGB","OKB","ETC",
        "APT","PEPE","PI","ENA","PUMP","ONDO","HTX","KCS","WLD","POL","HASH","TRUMP",
        "ALGO","FIL","ATOM","ARB","GT","VET","STRK","QNT","KAS","SKY","FLR","RENDER",
        "MORPHO","DASH","NEXO","SEI","IP","XDC","BONK","JUP","RAIN","FET","CAKE",
        "FTN","PENGU","VIRTUAL","AERO","OP","INJ","TIA","LDO","BDX","STX","TEL",
        "CRV","MYX","GRT","XTZ","DCR","IOTA","FLOKI","KAIA","SPX","PYTH","TWT","XPL",
        "ENS","CFX","GHO","SAND","SYRUP","SUN","FLOW","BTT","SOON","WIF","PENDLE",
        "HNT","THETA","NFT","GALA"
    ]

    # Format as pair
    symbols = [s + "/USDT" for s in symbols]

    # Only keep symbols that actually trade on Binance
    binance_pairs = get_usdt_pairs()
    return [s for s in symbols if s in binance_pairs]

def moving_average(series, length):
    return series.rolling(length).mean()


def compute_symbol_health_series(df: pd.DataFrame) -> pd.DataFrame:
    # If dataframe is empty, return empty
    if df.empty or len(df) < 2:
        return pd.DataFrame(index=df.index)

    close = df["close"]
    data = pd.DataFrame(index=df.index)

    # Short-term market structure
    # Only valid if enough history exists
    data["higher_high"] = (close > close.rolling(10).max().shift(1)).astype(float)

    # Returns for outperform calculations — natural NaN when insufficient data
    data["return_30d"] = close.pct_change(30)
    data["return_60d"] = close.pct_change(60)
    data["return_90d"] = close.pct_change(90)

    # Positive 30D return only where valid
    data["positive_30d"] = (data["return_30d"] > 0).astype(float)

    return data

def compute_market_health_timeseries(health_dict):
    # Merge symbols → MultiIndex columns: (symbol, feature)
    merged = pd.concat(health_dict.values(), axis=1, keys=health_dict.keys())

    result = pd.DataFrame(index=merged.index)

    # ---- Base metrics ----
    result["pct_higher_high"] = (
        merged.xs("higher_high", axis=1, level=1)
        .mean(axis=1, skipna=True) * 100
    )

    result["pct_positive_30d"] = (
        merged.xs("positive_30d", axis=1, level=1)
        .mean(axis=1, skipna=True) * 100
    )

    # ---- BTC returns ----
    btc_returns = merged.loc[:, ("BTC/USDT", ["return_30d", "return_60d", "return_90d"])]
    btc_returns.columns = ["btc_30d", "btc_60d", "btc_90d"]

    # ---- Coin returns ----
    ret30 = merged.xs("return_30d", axis=1, level=1)
    ret60 = merged.xs("return_60d", axis=1, level=1)
    ret90 = merged.xs("return_90d", axis=1, level=1)

    # Exclude BTC
    alt_cols = [c for c in ret30.columns if c != "BTC/USDT"]

    ret30_alt = ret30[alt_cols]
    ret60_alt = ret60[alt_cols]
    ret90_alt = ret90[alt_cols]

    # ---- Proper BTC outperform logic (NaN-preserving) ----

    # 30D
    diff30 = ret30_alt.subtract(btc_returns["btc_30d"], axis=0)
    outperform30 = (diff30 > 0).where(~diff30.isna(), np.nan)
    result["outperform_30d"] = outperform30.mean(axis=1, skipna=True) * 100

    # 60D
    diff60 = ret60_alt.subtract(btc_returns["btc_60d"], axis=0)
    outperform60 = (diff60 > 0).where(~diff60.isna(), np.nan)
    result["outperform_60d"] = outperform60.mean(axis=1, skipna=True) * 100

    # 90D
    diff90 = ret90_alt.subtract(btc_returns["btc_90d"], axis=0)
    outperform90 = (diff90 > 0).where(~diff90.isna(), np.nan)
    result["outperform_90d"] = outperform90.mean(axis=1, skipna=True) * 100

    return result

def create_table_health():
    create_sql = """
    CREATE TABLE IF NOT EXISTS market_health (
        date DATE PRIMARY KEY,
        pct_higher_high FLOAT,
        pct_positive_30d FLOAT,
        outperform_30d FLOAT,
        outperform_60d FLOAT,
        outperform_90d FLOAT
    );
    """

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(create_sql)
            conn.commit()


def insert_market_health(df: pd.DataFrame):
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:

            for date in df.index:
                row = df.loc[date]

                cur.execute(
                    """
                    INSERT INTO market_health (
                        date,
                        pct_higher_high,
                        pct_positive_30d,
                        outperform_30d,
                        outperform_60d,
                        outperform_90d
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date) DO UPDATE
                    SET 
                        pct_higher_high = EXCLUDED.pct_higher_high,
                        pct_positive_30d = EXCLUDED.pct_positive_30d,
                        outperform_30d = EXCLUDED.outperform_30d,
                        outperform_60d = EXCLUDED.outperform_60d,
                        outperform_90d = EXCLUDED.outperform_90d;
                    """,
                    (
                        date.date(),
                        float(row["pct_higher_high"]),
                        float(row["pct_positive_30d"]),
                        float(row["outperform_30d"]),
                        float(row["outperform_60d"]),
                        float(row["outperform_90d"])
                    )
                )

        conn.commit()


if __name__ == "__main__":
    DB_URL = ""

    symbols = get_symbols()
    print(f"Processing {len(symbols)} symbols...")

    health_history = {}

    for symbol in tqdm.tqdm(symbols, "Working..."):
        try:
            df = fetch_ohlcv_data(
                symbol=symbol,
                start_date="2018-01-01",
                end_date="2030-01-01",
                timeframe="1d",
                exchange_name="binance"
            )

            health_history[symbol] = compute_symbol_health_series(df)

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    # ---- Historical Market Health ----
    market_health_df = compute_market_health_timeseries(health_history)

    create_table_health()
    insert_market_health(market_health_df)

    print(market_health_df.tail())


''' )