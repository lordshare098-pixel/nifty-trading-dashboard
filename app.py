"""
Nifty Trading Dashboard (full version)
Tabs: Heatmap | Signals & Alerts | Stock Chart | Intraday VWAP | Backtest
Data: yfinance (free, ~15 min delayed). Sirf education ke liye.
Run:  pip install -r requirements.txt  ->  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

try:
from streamlit_autorefresh import st_autorefresh
except Exception:
st_autorefresh = None

st.set_page_config(page_title="Nifty Dashboard", layout="wide")

SECTORS = {
"Banks": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN", "INDUSINDBK"],
"Financials": ["BAJFINANCE", "BAJAJFINSV", "SBILIFE", "HDFCLIFE"],
"IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM"],
"Energy": ["RELIANCE", "ONGC", "POWERGRID", "NTPC", "COALINDIA", "BPCL"],
"Auto": ["TATAMOTORS", "MARUTI", "M&M", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO"],
"FMCG": ["ITC", "HINDUNILVR", "NESTLEIND", "BRITANNIA", "TITAN"],
"Pharma": ["SUNPHARMA", "CIPLA", "DRREDDY", "APOLLOHOSP"],
"Metals & Infra": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "LT", "ULTRACEMCO", "ADANIPORTS"],
"Telecom": ["BHARTIARTL"],
}
T2S = {f"{s}.NS": sec for sec, lst in SECTORS.items() for s in lst}
TICKERS = list(T2S)

------------------------------------------------------------ helpers

def flat(df):
if isinstance(df.columns, pd.MultiIndex):
df.columns = df.columns.get_level_values(0)
return df

def rsi(c, n=14):
d = c.diff()
up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
return 100 - 100 / (1 + up / dn)

def add_ind(df):
df = df.copy()
c = df["Close"]
df["EMA20"] = c.ewm(span=20, adjust=False).mean()
df["EMA50"] = c.ewm(span=50, adjust=False).mean()
df["RSI"] = rsi(c)
df["MACD"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
df["MACDs"] = df["MACD"].ewm(span=9, adjust=False).mean()
tr = pd.concat([df["High"] - df["Low"],
(df["High"] - c.shift()).abs(),
(df["Low"] - c.shift()).abs()], axis=1).max(axis=1)
df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
df["Score"] = (np.where(c > df["EMA20"], 1, -1)
+ np.where(df["EMA20"] > df["EMA50"], 1, -1)
+ np.where(df["RSI"] > 55, 1, np.where(df["RSI"] < 45, -1, 0))
+ np.where(df["MACD"] > df["MACDs"], 1, -1))
return df

def label(score):
return "BULLISH" if score >= 3 else "BEARISH" if score <= -3 else "NEUTRAL"

@st.cache_data(ttl=300, show_spinner="Data aa raha hai...")
def load_all(tickers):
return yf.download(list(tickers), period="1y", interval="1d", group_by="ticker",
auto_adjust=True, progress=False, threads=True)

@st.cache_data(ttl=300)
def load_one(ticker, period, interval):
return flat(yf.download(ticker, period=period, interval=interval,
auto_adjust=True, progress=False))

def backtest(df, cost=0.0005):
df = add_ind(df.dropna())
pos = ((df["EMA20"] > df["EMA50"]) & (df["RSI"] > 50)).astype(int).shift(1).fillna(0)
ret = df["Close"].pct_change().fillna(0) * pos
fee = pos.diff().abs().fillna(0) * cost
net = ret - fee
eq = (1 + net).cumprod()
bh = (1 + df["Close"].pct_change().fillna(0)).cumprod()
tid = ((pos.diff() == 1).cumsum()).where(pos == 1)
trades = (1 + net).groupby(tid).prod() - 1
stats = {
"Strategy return %": (eq.iloc[-1] - 1) * 100,
"Buy & Hold %": (bh.iloc[-1] - 1) * 100,
"Max drawdown %": (eq / eq.cummax() - 1).min() * 100,
"Trades": int(len(trades)),
"Win rate %": float((trades > 0).mean() * 100) if len(trades) else 0.0,
}
return eq, bh, stats

------------------------------------------------------------ sidebar

st.sidebar.header("Settings")
sector_pick = st.sidebar.multiselect("Sectors", list(SECTORS), default=list(SECTORS))
auto = st.sidebar.checkbox("Auto refresh (60s)", value=False)
if auto and st_autorefresh:
st_autorefresh(interval=60_000, key="auto")
elif auto:
st.sidebar.warning("pip install streamlit-autorefresh")
if st.sidebar.button("🔄 Refresh now"):
st.cache_data.clear()

st.title("📊 Nifty Trading Dashboard")
st.caption("Free delayed data. Signals probability dikhate hain, guarantee nahi. Financial advice nahi.")

------------------------------------------------------------ compute

raw = load_all(tuple(TICKERS))
rows, frames = [], {}
for t in TICKERS:
if T2S[t] not in sector_pick:
continue
try:
df = add_ind(raw[t].dropna())
frames[t] = df
last, prev = df.iloc[-1], df.iloc[-2]
crossed = ""
if prev["EMA20"] <= prev["EMA50"] and last["EMA20"] > last["EMA50"]:
crossed = "Golden cross ↑"
elif prev["EMA20"] >= prev["EMA50"] and last["EMA20"] < last["EMA50"]:
crossed = "Death cross ↓"
rows.append({
"Stock": t.replace(".NS", ""), "Sector": T2S[t],
"Price": round(last["Close"], 2),
"Change %": round((last["Close"] / prev["Close"] - 1) * 100, 2),
"RSI": round(last["RSI"], 1), "Score": int(last["Score"]),
"Signal": label(last["Score"]),
"Stop-loss (1.5 ATR)": round(last["Close"] - 1.5 * last["ATR"], 2),
"Event": crossed, "Size": 1,
})
except Exception:
continue
res = pd.DataFrame(rows)
if res.empty:
st.error("Data nahi mila. Internet check karo aur Refresh dabao.")
st.stop()

idx = add_ind(load_one("^NSEI", "1y", "1d").dropna())
up, down = int((res["Change %"] > 0).sum()), int((res["Change %"] < 0).sum())
breadth = up / max(up + down, 1)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Nifty", f"{idx['Close'].iloc[-1]:,.0f}",
f"{(idx['Close'].iloc[-1] / idx['Close'].iloc[-2] - 1) * 100:.2f}%")
m2.metric("Nifty Signal", label(idx["Score"].iloc[-1]), f"score {int(idx['Score'].iloc[-1])}/4")
m3.metric("Advancing", up)
m4.metric("Declining", down)
st.progress(breadth, text=f"Market breadth: {breadth * 100:.0f}% stocks upar")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
["🟩 Heatmap", "🔔 Signals", "📈 Chart", "⏱ Intraday", "🧪 Backtest"])

------------------------------------------------------------ heatmap

with tab1:
fig = px.treemap(res, path=["Sector", "Stock"], values="Size", color="Change %",
color_continuous_scale=["#d6284a", "#1c1c1c", "#13c27a"],
range_color=[-3, 3], hover_data=["Price", "Signal", "RSI"])
fig.update_traces(texttemplate="%{label}<br>%{color:.2f}%")
fig.update_layout(height=560, margin=dict(t=5, l=0, r=0, b=0))
st.plotly_chart(fig, use_container_width=True)

sec = res.groupby("Sector")["Change %"].mean().sort_values()  
st.plotly_chart(px.bar(sec, orientation="h", color=sec,  
                       color_continuous_scale=["#d6284a", "#888", "#13c27a"],  
                       range_color=[-3, 3], title="Sector average % change")  
                .update_layout(height=320, coloraxis_showscale=False, showlegend=False),  
                use_container_width=True)

------------------------------------------------------------ signals

with tab2:
st.subheader("Alerts")
a1, a2 = st.columns(2)
with a1:
st.markdown("🟢 Strong Bullish")
st.write(", ".join(res[res["Score"] == 4]["Stock"]) or "Koi nahi")
st.markdown("✨ Golden cross aaj")
st.write(", ".join(res[res["Event"].str.startswith("Golden")]["Stock"]) or "Koi nahi")
with a2:
st.markdown("🔴 Strong Bearish")
st.write(", ".join(res[res["Score"] == -4]["Stock"]) or "Koi nahi")
st.markdown("💀 Death cross aaj")
st.write(", ".join(res[res["Event"].str.startswith("Death")]["Stock"]) or "Koi nahi")

view = st.radio("Filter", ["All", "BULLISH", "BEARISH", "NEUTRAL"], horizontal=True)  
tbl = res if view == "All" else res[res["Signal"] == view]  
st.dataframe(tbl.drop(columns="Size").sort_values("Score", ascending=False),  
             use_container_width=True, hide_index=True)  
with st.expander("Signal kaise banta hai?"):  
    st.write("""

Har stock ko 4 checks pe +1 / -1 milta hai:

1. Price > EMA20 (short trend)
2. EMA20 > EMA50 (trend direction)
3. RSI > 55 (+1) ya < 45 (-1)
4. MACD > Signal line

Score +3/+4 = BULLISH, -3/-4 = BEARISH, baaki NEUTRAL.
Stop-loss = Price - 1.5 x ATR (volatility based, long trade ke liye).
""")

------------------------------------------------------------ chart

names = list(res["Stock"])
with tab3:
pick = st.selectbox("Stock", names, key="chart_pick")
d = frames[pick + ".NS"].tail(150)
fig = go.Figure(go.Candlestick(x=d.index, open=d["Open"], high=d["High"],
low=d["Low"], close=d["Close"], name="Price"))
fig.add_scatter(x=d.index, y=d["EMA20"], name="EMA20", line=dict(width=1.5))
fig.add_scatter(x=d.index, y=d["EMA50"], name="EMA50", line=dict(width=1.5))
fig.update_layout(height=480, xaxis_rangeslider_visible=False, margin=dict(t=10, b=0))
st.plotly_chart(fig, use_container_width=True)
r = go.Figure(go.Scatter(x=d.index, y=d["RSI"], name="RSI"))
r.add_hline(y=70, line_dash="dot")
r.add_hline(y=30, line_dash="dot")
r.update_layout(height=200, margin=dict(t=10, b=0), title="RSI (14)")
st.plotly_chart(r, use_container_width=True)
row = res[res["Stock"] == pick].iloc[0]
st.info(f"{pick}: {row['Signal']} (score {row['Score']}) | "
f"Price {row['Price']} | Stop-loss ≈ {row['Stop-loss (1.5 ATR)']}")

------------------------------------------------------------ intraday

with tab4:
c1, c2 = st.columns(2)
ipick = c1.selectbox("Stock", names, key="intra_pick")
tf = c2.selectbox("Timeframe", ["5m", "15m"], index=1)
d = load_one(ipick + ".NS", "5d", tf).dropna()
if d.empty:
st.warning("Intraday data abhi nahi mila (market band ho sakta hai).")
else:
d["Day"] = d.index.date
tp = (d["High"] + d["Low"] + d["Close"]) / 3
d["VWAP"] = (tp * d["Volume"]).groupby(d["Day"]).cumsum() / \
d["Volume"].groupby(d["Day"]).cumsum()
d["EMA9"] = d["Close"].ewm(span=9, adjust=False).mean()
fig = go.Figure(go.Candlestick(x=d.index, open=d["Open"], high=d["High"],
low=d["Low"], close=d["Close"], name="Price"))
fig.add_scatter(x=d.index, y=d["VWAP"], name="VWAP", line=dict(width=2))
fig.add_scatter(x=d.index, y=d["EMA9"], name="EMA9", line=dict(width=1))
fig.update_layout(height=480, xaxis_rangeslider_visible=False, margin=dict(t=10, b=0))
st.plotly_chart(fig, use_container_width=True)
last = d.iloc[-1]
bias = "BULLISH bias (price VWAP ke upar)" if last["Close"] > last["VWAP"] \
else "BEARISH bias (price VWAP ke niche)"
st.info(f"{ipick}: {bias}. Price {last['Close']:.2f} | VWAP {last['VWAP']:.2f}")

------------------------------------------------------------ backtest

with tab5:
st.write("Rule: EMA20 > EMA50 aur RSI > 50 to long, warna cash. "
"Entry agle din (lookahead nahi), 0.05% cost.")
bpick = st.selectbox("Stock / Index", ["NIFTY 50"] + names, key="bt_pick")
period = st.selectbox("Period", ["1y", "2y", "5y"], index=1)
sym = "^NSEI" if bpick == "NIFTY 50" else bpick + ".NS"
bd = load_one(sym, period, "1d").dropna()
if len(bd) < 80:
st.warning("Kam data hai.")
else:
eq, bh, s = backtest(bd)
cols = st.columns(5)
for col, (k, v) in zip(cols, s.items()):
col.metric(k, f"{v:.1f}" if isinstance(v, float) else v)
fig = go.Figure()
fig.add_scatter(x=eq.index, y=eq, name="Strategy")
fig.add_scatter(x=bh.index, y=bh, name="Buy & Hold")
fig.update_layout(height=380, margin=dict(t=10, b=0))
st.plotly_chart(fig, use_container_width=True)
st.caption("Past performance future ki guarantee nahi. Live jaane se pehle paper trade karo.")
