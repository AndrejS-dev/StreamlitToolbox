import streamlit as st
import psycopg
import pandas as pd
import plotly.graph_objects as go

DB_URL = st.secrets["DB_URL"]
st.set_page_config("Crypto Breadth", layout="wide")
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

st.title("Crypto Breadth")
st.text("""
This indicator calculates the percentage of cryptocurrencies trading above their 30, 60, and 90-day Simple Moving Averages (SMA). 
It fetches OHLC data for major USDT pairs, evaluates whether each asset is above or below its respective SMA, and then averages these values to create breadth indexes. 
Higher values indicate broad market strength, while lower values signal widespread weakness.
""")

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
    title="",
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

st.markdown("---")
with st.expander("Full DataFrame"):
    st.dataframe(df)

with st.expander("Source Code"):
    st.code(
'''
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import psycopg

def fetch_ohlcv_data(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str = '1d',
    exchange_name: str = 'binance'
) -> pd.DataFrame:

    # --- Initialize exchange ---
    try:
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class()
    except AttributeError:
        raise ValueError(f"Exchange '{exchange_name}' is not supported by CCXT.")

    # --- Convert dates to timestamps ---
    since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

    ohlcv = []

    # --- Paginated data fetching ---
    while since < end_timestamp:
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not data:
                break

            ohlcv.extend(data)
            since = data[-1][0] + 1  # Move to next batch

        except Exception as e:
            print(f"Error fetching data for {symbol} on {exchange_name}: {e}")
            break

    # --- Convert to DataFrame ---
    if not ohlcv:
        raise ValueError(f"No data returned for {symbol} on {exchange_name}.")

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('time', inplace=True)

    # --- Return formatted DataFrame ---
    return df[['open', 'high', 'low', 'close', 'volume']]

def get_usdt_pairs(exchange_name="binance"):
    exchange = getattr(ccxt, exchange_name)()
    markets = exchange.load_markets()
    return [s for s in markets.keys() if s.endswith("/USDT")]

def get_symbols():
    symbols = ["BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "ZEC", "HYPE", "BCH", "LINK", "LEO", "XLM", "LTC", "XMR",
               "HBAR", "AVAX", "SUI", "SHIB", "UNI", "TON", "DOT", "CRO", "WLFI", "CC", "M", "MNT", "TAO", "NEAR", "ICP", "ASTER",
               "AAVE", "BGB", "OKB", "ETC", "APT", "PEPE", "PI", "ENA", "PUMP", "ONDO", "HTX", "KCS", "WLD", "POL", "HASH", "TRUMP",
               "ALGO", "FIL", "ATOM", "ARB", "GT", "VET", "STRK", "QNT", "KAS", "SKY", "FLR", "RENDER", "MORPHO", "DASH", "NEXO",
               "SEI", "IP", "XDC", "BONK", "JUP", "RAIN", "FET", "CAKE", "FTN", "PENGU", "VIRTUAL", "AERO", "OP", "INJ", "TIA",
               "LDO", "BDX", "STX", "TEL", "CRV", "MYX", "GRT", "XTZ", "DCR", "IOTA", "FLOKI", "KAIA", "SPX", "PYTH", "TWT", "XPL",
               "ENS", "CFX", "GHO", "SAND", "SYRUP", "SUN", "FLOW", "BTT", "SOON", "WIF", "PENDLE", "HNT", "THETA", "NFT", "GALA",]
    
    binance_pairs = get_usdt_pairs()
    
    for i in range(len(symbols)):
        symbols[i] = symbols[i] + "/USDT"

    binance_symbols = [symbol for symbol in symbols if symbol in binance_pairs] 
    return binance_symbols

def sma(series: pd.Series, length: int) -> pd.Series:
    sma_raw = series.rolling(window=length, min_periods=length).mean()
    return sma_raw.fillna(0)

def create_table():
    """Create breadth table if it does not exist."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS breadth (
        date DATE PRIMARY KEY,
        sma30 FLOAT,
        sma60 FLOAT,
        sma90 FLOAT
    );
    """

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(create_sql)
            conn.commit()


def insert_breadth(df30: pd.DataFrame, df60: pd.DataFrame, df90: pd.DataFrame):
    """Insert breadth index values into database."""
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            
            for date in df30.index:
                sma30 = float(df30.loc[date, "index"])
                sma60 = float(df60.loc[date, "index"])
                sma90 = float(df90.loc[date, "index"])

                cur.execute(
                    """
                    INSERT INTO breadth (date, sma30, sma60, sma90)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (date) DO UPDATE
                    SET sma30 = EXCLUDED.sma30,
                        sma60 = EXCLUDED.sma60,
                        sma90 = EXCLUDED.sma90;
                    """,
                    (date.date(), sma30, sma60, sma90)
                )

        conn.commit()


if __name__ == "__main__":
    DB_URL = "your_database_connection_string"

    symbols = get_symbols()
    
    breadth_sma_90 = pd.DataFrame()
    breadth_sma_60 = pd.DataFrame()
    breadth_sma_30 = pd.DataFrame()

    for symbol in symbols:
        try:
            df = fetch_ohlcv_data(symbol, "2025-01-01", "2030-01-01", "1d")

            df['sma_90'] = sma(df['close'], 90).fillna(0)
            df['sma_60'] = sma(df['close'], 60).fillna(0)            
            df['sma_30'] = sma(df['close'], 30).fillna(0)

            df['sma_90_index'] = np.where(df['close'] > df["sma_90"], np.where((df["sma_90"] == 0), 0, 1), 0)
            df['sma_60_index'] = np.where(df['close'] > df["sma_60"], np.where((df["sma_60"] == 0), 0, 1), 0)
            df['sma_30_index'] = np.where(df['close'] > df["sma_30"], np.where((df["sma_30"] == 0), 0, 1), 0)

            breadth_sma_90[symbol] = df['sma_90_index']
            breadth_sma_60[symbol] = df['sma_60_index']
            breadth_sma_30[symbol] = df['sma_30_index']
        except Exception as e:
            continue

    breadth_sma_90["index"] = breadth_sma_90.mean(axis=1)
    breadth_sma_60["index"] = breadth_sma_60.mean(axis=1)
    breadth_sma_30["index"] = breadth_sma_30.mean(axis=1)

    create_table()
    insert_breadth(breadth_sma_30, breadth_sma_60, breadth_sma_90)
    ''')