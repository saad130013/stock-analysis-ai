import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
from prophet import Prophet
import os
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

@st.cache_data
def load_all_symbols():
    try:
        df = pd.read_csv("saudi_stocks.csv")
        return dict(zip(df["اسم السهم"], df["الرمز"]))
    except Exception as e:
        st.error("فشل في تحميل رموز الأسهم: تأكد من وجود ملف saudi_stocks.csv")
        return {}

symbols = load_all_symbols()
recommendation_log_file = "recommendation_log.csv"

# 📈 صفحة ملخص السوق
@st.cache_data
def get_market_summary():
    results = []
    for name, symbol in symbols.items():
        try:
            data = yf.download(symbol, period="5d", interval="1d")
            if not data.empty:
                latest = data.iloc[-1]
                prev = data.iloc[-2] if len(data) > 1 else latest
                change = latest['Close'] - prev['Close']
                percent = (change / prev['Close']) * 100 if prev['Close'] != 0 else 0
                results.append({
                    "الرمز": symbol,
                    "السهم": name,
                    "السعر": round(latest['Close'], 2),
                    "التغير %": round(percent, 2),
                    "السيولة": round(latest['Volume'], 0)
                })
        except:
            continue
    return pd.DataFrame(results)

# عرض الواجهة
st.set_page_config(page_title="خبير الأسهم الفني", layout="wide")
st.title("📊 نظام خبير الأسهم الفني السعودي")
st.markdown("نظام شامل لتحليل الأسهم: مؤشرات فنية، نماذج، تنبؤات، ذكاء اصطناعي، إشعارات وتوصيات")

option = st.radio("اختر وضع التشغيل", ("📊 تحليل سهم محدد", "📈 ملخص السوق"))

if option == "📈 ملخص السوق":
    st.subheader("📋 ملخص السوق السعودي")
    market_data = get_market_summary()

    if not market_data.empty:
        total_liquidity = market_data["السيولة"].sum()
        up = market_data[market_data["التغير %"] > 0].shape[0]
        down = market_data[market_data["التغير %"] < 0].shape[0]
        unchanged = market_data.shape[0] - up - down

        st.metric("📈 عدد الأسهم المرتفعة", up)
        st.metric("📉 عدد الأسهم المنخفضة", down)
        st.metric("➖ بدون تغيير", unchanged)
        st.metric("💰 السيولة الإجمالية", f"{total_liquidity:,.0f}")

        top_recos = market_data.sort_values("التغير %", ascending=False).head(5)
        st.markdown("### 🌟 أقوى 5 فرص اليوم")
        st.dataframe(top_recos)
    else:
        st.info("لا توجد بيانات متاحة لعرض ملخص السوق")

elif option == "📊 تحليل سهم محدد":
    selected_stock = st.selectbox("اختر السهم", list(symbols.keys()))
    analyze_stock(selected_stock, symbols[selected_stock])
