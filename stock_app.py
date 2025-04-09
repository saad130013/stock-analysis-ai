{\rtf1\ansi\ansicpg1252\cocoartf2580
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset178 GeezaPro;\f2\fnil\fcharset0 AppleColorEmoji;
\f3\fnil\fcharset0 LucidaGrande;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww28600\viewh15820\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import yfinance as yf\
import pandas as pd\
import pandas_ta as ta\
import streamlit as st\
from datetime import datetime, timedelta\
import numpy as np\
from prophet import Prophet\
import os\
import plotly.graph_objs as go\
from sklearn.ensemble import RandomForestRegressor\
from tensorflow.keras.models import Sequential\
from tensorflow.keras.layers import LSTM, Dense\
import matplotlib.pyplot as plt\
\
# Load all Saudi stock symbols dynamically\
@st.cache_data\
def load_all_symbols():\
    # Fetch Saudi stock symbols from a reliable source or use a predefined CSV\
    try:\
        df = pd.read_csv("saudi_stocks.csv")  # Ensure this file exists or fetch dynamically\
        return dict(zip(df["
\f1 \'c7\'d3\'e3
\f0  
\f1 \'c7\'e1\'d3\'e5\'e3
\f0 "], df["
\f1 \'c7\'e1\'d1\'e3\'d2
\f0 "]))\
    except FileNotFoundError:\
        st.error("File 'saudi_stocks.csv' not found. Please upload a valid file.")\
        return \{\}\
\
symbols = load_all_symbols()\
\
recommendation_log_file = "recommendation_log.csv"\
\
def save_recommendation(symbol, recommendation):\
    now = datetime.now().strftime("%Y-%m-%d %H:%M")\
    record = pd.DataFrame([[now, symbol, recommendation]], columns=["
\f1 \'c7\'e1\'ca\'c7\'d1\'ed\'ce
\f0 ", "
\f1 \'c7\'e1\'d3\'e5\'e3
\f0 ", "
\f1 \'c7\'e1\'ca\'e6\'d5\'ed\'c9
\f0 "])\
    if os.path.exists(recommendation_log_file):\
        record.to_csv(recommendation_log_file, mode='a', header=False, index=False)\
    else:\
        record.to_csv(recommendation_log_file, index=False)\
\
def detect_chart_patterns(data):\
    patterns = []\
    closes = data["Close"].tail(100)\
    highs = data["High"].tail(100)\
    lows = data["Low"].tail(100)\
    if closes.iloc[-1] > closes.mean() and lows.min() < closes.mean() * 0.95:\
        patterns.append("
\f2 \uc0\u9749 
\f0  
\f1 \'e4\'e3\'e6\'d0\'cc
\f0  
\f1 \'df\'e6\'c8
\f0  
\f1 \'e6\'da\'d1\'e6\'c9
\f0  
\f1 \'e3\'cd\'ca\'e3\'e1
\f0 ")\
    if highs.max() - highs.min() < 0.03 * highs.max():\
        patterns.append("
\f2 \uc0\u55357 \u56634 
\f0  
\f1 \'e4\'e3\'e6\'d0\'cc
\f0  
\f1 \'e3\'cb\'e1\'cb
\f0  
\f1 \'d5\'c7\'da\'cf
\f0  
\f1 \'e3\'cd\'ca\'e3\'e1
\f0 ")\
    if lows.max() - lows.min() < 0.03 * lows.max():\
        patterns.append("
\f2 \uc0\u55357 \u56635 
\f0  
\f1 \'e4\'e3\'e6\'d0\'cc
\f0  
\f1 \'e3\'cb\'e1\'cb
\f0  
\f1 \'e5\'c7\'c8\'d8
\f0  
\f1 \'e3\'cd\'ca\'e3\'e1
\f0 ")\
    return patterns\
\
def predict_with_lstm(data):\
    # Prepare data for LSTM\
    data = data["Close"].values.reshape(-1, 1)\
    train_size = int(len(data) * 0.8)\
    train, test = data[:train_size], data[train_size:]\
\
    def create_dataset(dataset, look_back=60):\
        X, y = [], []\
        for i in range(look_back, len(dataset)):\
            X.append(dataset[i - look_back:i, 0])\
            y.append(dataset[i, 0])\
        return np.array(X), np.array(y)\
\
    look_back = 60\
    X_train, y_train = create_dataset(train, look_back)\
    X_test, y_test = create_dataset(test, look_back)\
\
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))\
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))\
\
    # Build LSTM model\
    model = Sequential([\
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),\
        LSTM(50, return_sequences=False),\
        Dense(25),\
        Dense(1)\
    ])\
    model.compile(optimizer='adam', loss='mean_squared_error')\
    model.fit(X_train, y_train, batch_size=1, epochs=1)\
\
    predictions = model.predict(X_test)\
    return predictions.flatten()\
\
def analyze_stock(name, symbol):\
    data = yf.download(symbol, period="1y", interval="1d")\
    data.dropna(inplace=True)\
\
    # Calculate technical indicators\
    data["RSI"] = ta.rsi(data["Close"], length=14)\
    macd_data = ta.macd(data["Close"])\
    if macd_data is not None and not macd_data.empty:\
        data["MACD"] = macd_data.iloc[:, 0]\
        data["MACD_signal"] = macd_data.iloc[:, 1]\
    else:\
        data["MACD"] = np.nan\
        data["MACD_signal"] = np.nan\
    data["MFI"] = ta.mfi(data["High"], data["Low"], data["Close"], data["Volume"])\
    data["OBV"] = ta.obv(data["Close"], data["Volume"])\
    data["SMA_50"] = ta.sma(data["Close"], length=50)\
    data["SMA_200"] = ta.sma(data["Close"], length=200)\
\
    latest_close = data["Close"].iloc[-1]\
    support = round(data["Close"].rolling(window=20).min().iloc[-1], 2)\
    resistance = round(data["Close"].rolling(window=20).max().iloc[-1], 2)\
\
    rsi = data["RSI"].iloc[-1]\
    macd = data["MACD"].iloc[-1]\
    macd_signal = data["MACD_signal"].iloc[-1]\
    mfi = data["MFI"].iloc[-1]\
    obv_change = data["OBV"].iloc[-1] - data["OBV"].iloc[-2]\
    sma_50 = data["SMA_50"].iloc[-1]\
    sma_200 = data["SMA_200"].iloc[-1]\
\
    st.subheader(f"
\f2 \uc0\u55357 \u56524 
\f0  
\f1 \'ca\'cd\'e1\'ed\'e1
\f0  
\f1 \'d3\'e5\'e3
\f0  \{name\}")\
    st.write(f"**
\f2 \uc0\u55357 \u56517 
\f0  
\f1 \'c7\'e1\'ca\'c7\'d1\'ed\'ce
\f0 :** \{datetime.today().strftime('%Y-%m-%d %H:%M')\}")\
    st.write(f"**
\f2 \uc0\u55357 \u56521 
\f0  
\f1 \'c7\'e1\'d3\'da\'d1
\f0  
\f1 \'c7\'e1\'cd\'c7\'e1\'ed
\f0 :** \{latest_close:.2f\} 
\f1 \'d1\'ed\'c7\'e1
\f0 ")\
\
    st.plotly_chart(go.Figure(data=[\
        go.Candlestick(\
            x=data.index,\
            open=data['Open'],\
            high=data['High'],\
            low=data['Low'],\
            close=data['Close'],\
            name='
\f1 \'d4\'e3\'e6\'da
\f0 '\
        )\
    ]))\
\
    st.markdown("---")\
    st.markdown("### 
\f2 \uc0\u55358 \u56800 
\f0  
\f1 \'c7\'e1\'e3\'c4\'d4\'d1\'c7\'ca
\f0  
\f1 \'c7\'e1\'dd\'e4\'ed\'c9
\f0 ")\
    st.write(f"**RSI:** \{rsi:.2f\}")\
    st.write(f"**MACD:** \{macd:.2f\}, **
\f1 \'c7\'e1\'c5\'d4\'c7\'d1\'c9
\f0 :** \{macd_signal:.2f\}")\
    st.write(f"**MFI:** \{mfi:.2f\}")\
    st.write(f"**OBV:** \{'
\f3 \uc0\u8593 
\f0 ' if obv_change > 0 else '
\f3 \uc0\u8595 
\f0 '\}")\
    st.write(f"**SMA-50:** \{sma_50:.2f\} | **SMA-200:** \{sma_200:.2f\}")\
    st.write(f"**
\f1 \'c7\'e1\'cf\'da\'e3
\f0 :** \{support\} | **
\f1 \'c7\'e1\'e3\'de\'c7\'e6\'e3\'c9
\f0 :** \{resistance\}")\
\
    st.markdown("### 
\f2 \uc0\u9989 
\f0  
\f1 \'c7\'e1\'ca\'e6\'d5\'ed\'c9
\f0 ")\
    recommendation = "
\f1 \'ca\'d1\'de\'c8
\f0  \'96 
\f1 \'e1\'c7
\f0  
\f1 \'ca\'e6\'cc\'cf
\f0  
\f1 \'c5\'d4\'c7\'d1\'c9
\f0  
\f1 \'e6\'c7\'d6\'cd\'c9
\f0 "\
    if rsi < 30 and macd > macd_signal:\
        recommendation = "
\f2 \uc0\u55357 \u56549 
\f0  
\f1 \'d4\'d1\'c7\'c1
\f0  
\f1 \'e3\'cd\'ca\'e3\'e1
\f0 "\
        st.success(recommendation)\
    elif rsi > 70 or macd < macd_signal:\
        recommendation = "
\f2 \uc0\u9888 \u65039 
\f0  
\f1 \'c7\'cd\'ca\'e3\'c7\'e1
\f0  
\f1 \'e5\'c8\'e6\'d8
\f0 "\
        st.warning(recommendation)\
    else:\
        st.info(recommendation)\
\
    save_recommendation(name, recommendation)\
\
    st.markdown("### 
\f2 \uc0\u55357 \u56528 
\f0  
\f1 \'c7\'e1\'e4\'e3\'c7\'d0\'cc
\f0  
\f1 \'c7\'e1\'dd\'e4\'ed\'c9
\f0 :")\
    patterns = detect_chart_patterns(data)\
    if patterns:\
        for pattern in patterns:\
            st.warning(pattern)\
    else:\
        st.info("
\f1 \'e1\'c7
\f0  
\f1 \'ca\'e6\'cc\'cf
\f0  
\f1 \'e4\'e3\'c7\'d0\'cc
\f0  
\f1 \'cd\'c7\'e1\'ed\'c7\'f0
\f0 ")\
\
    st.markdown("### 
\f2 \uc0\u55358 \u56598 
\f0  
\f1 \'ca\'cd\'e1\'ed\'e1
\f0  
\f1 \'c7\'e1\'d0\'df\'c7\'c1
\f0  
\f1 \'c7\'e1\'c7\'d5\'d8\'e4\'c7\'da\'ed
\f0  
\f1 \'e1\'e1\'e4\'e3\'d8
\f0 :")\
    lstm_predictions = predict_with_lstm(data)\
    st.write("**
\f1 \'ca\'e6\'de\'da\'c7\'ca
\f0  
\f1 \'c3\'d3\'da\'c7\'d1
\f0  
\f1 \'c7\'e1\'c5\'db\'e1\'c7\'de
\f0  
\f1 \'c7\'e1\'e3\'d3\'ca\'de\'c8\'e1\'ed\'c9
\f0  
\f1 \'c8\'c7\'d3\'ca\'ce\'cf\'c7\'e3
\f0  LSTM:**")\
    st.line_chart(lstm_predictions)\
\
    st.markdown("### 
\f2 \uc0\u55357 \u56622 
\f0  
\f1 \'ca\'e6\'de\'da\'c7\'ca
\f0  5 
\f1 \'c3\'ed\'c7\'e3
\f0 ")\
    df_prophet = data.reset_index()[["Date", "Close"]].rename(columns=\{"Date": "ds", "Close": "y"\})\
    model = Prophet(daily_seasonality=True)\
    model.fit(df_prophet)\
    future = model.make_future_dataframe(periods=5)\
    forecast = model.predict(future)\
    forecast = forecast[["ds", "yhat"]].tail(5).rename(columns=\{"ds": "
\f1 \'c7\'e1\'ca\'c7\'d1\'ed\'ce
\f0 ", "yhat": "
\f1 \'c7\'e1\'d3\'da\'d1
\f0  
\f1 \'c7\'e1\'e3\'ca\'e6\'de\'da
\f0 "\})\
    st.line_chart(forecast.set_index("
\f1 \'c7\'e1\'ca\'c7\'d1\'ed\'ce
\f0 "))\
\
    if os.path.exists(recommendation_log_file):\
        st.markdown("### 
\f2 \uc0\u55357 \u56770 \u65039 
\f0  
\f1 \'d3\'cc\'e1
\f0  
\f1 \'c7\'e1\'ca\'e6\'d5\'ed\'c7\'ca
\f0 ")\
        log_df = pd.read_csv(recommendation_log_file)\
        st.dataframe(log_df.tail(10))\
\
st.set_page_config(page_title="
\f1 \'ce\'c8\'ed\'d1
\f0  
\f1 \'c7\'e1\'c3\'d3\'e5\'e3
\f0  
\f1 \'c7\'e1\'dd\'e4\'ed
\f0 ", layout="wide")\
st.title("
\f2 \uc0\u55357 \u56522 
\f0  
\f1 \'e4\'d9\'c7\'e3
\f0  
\f1 \'ce\'c8\'ed\'d1
\f0  
\f1 \'c7\'e1\'c3\'d3\'e5\'e3
\f0  
\f1 \'c7\'e1\'dd\'e4\'ed
\f0  
\f1 \'c7\'e1\'d3\'da\'e6\'cf\'ed
\f0 ")\
st.markdown("
\f1 \'e4\'d9\'c7\'e3
\f0  
\f1 \'d4\'c7\'e3\'e1
\f0  
\f1 \'e1\'ca\'cd\'e1\'ed\'e1
\f0  
\f1 \'c7\'e1\'c3\'d3\'e5\'e3
\f0 : 
\f1 \'e3\'c4\'d4\'d1\'c7\'ca
\f0  
\f1 \'dd\'e4\'ed\'c9\'a1
\f0  
\f1 \'e4\'e3\'c7\'d0\'cc\'a1
\f0  
\f1 \'ca\'e4\'c8\'c4\'c7\'ca\'a1
\f0  
\f1 \'d0\'df\'c7\'c1
\f0  
\f1 \'c7\'d5\'d8\'e4\'c7\'da\'ed\'a1
\f0  
\f1 \'c5\'d4\'da\'c7\'d1\'c7\'ca
\f0  
\f1 \'e6\'ca\'e6\'d5\'ed\'c7\'ca
\f0 ")\
option = st.radio("
\f1 \'c7\'ce\'ca\'d1
\f0  
\f1 \'e6\'d6\'da
\f0  
\f1 \'c7\'e1\'ca\'d4\'db\'ed\'e1
\f0 ", ("
\f1 \'ca\'cd\'e1\'ed\'e1
\f0  
\f1 \'d3\'e5\'e3
\f0  
\f1 \'e3\'cd\'cf\'cf
\f0 ",))\
\
if option == "
\f1 \'ca\'cd\'e1\'ed\'e1
\f0  
\f1 \'d3\'e5\'e3
\f0  
\f1 \'e3\'cd\'cf\'cf
\f0 ":\
    selected_stock = st.selectbox("
\f1 \'c7\'ce\'ca\'d1
\f0  
\f1 \'c7\'e1\'d3\'e5\'e3
\f0 ", list(symbols.keys()))\
    analyze_stock(selected_stock, symbols[selected_stock])}