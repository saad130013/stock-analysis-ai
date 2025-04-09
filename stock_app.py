import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
from prophet import Prophet
import os
import plotly.graph_objs as go
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2

model_ai = ResNet50(weights='imagenet')

@st.cache_data
def load_all_symbols():
    df = pd.read_csv("saudi_stocks.csv")
    return dict(zip(df["اسم السهم"], df["الرمز"]))

symbols = load_all_symbols()
recommendation_log_file = "recommendation_log.csv"

def save_recommendation(symbol, recommendation):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    record = pd.DataFrame([[now, symbol, recommendation]], columns=["التاريخ", "السهم", "التوصية"])
    if os.path.exists(recommendation_log_file):
        record.to_csv(recommendation_log_file, mode='a', header=False, index=False)
    else:
        record.to_csv(recommendation_log_file, index=False)

def send_alert(title, message):
    pass

def detect_chart_patterns(data):
    patterns = []
    closes = data["Close"].tail(100)
    highs = data["High"].tail(100)
    lows = data["Low"].tail(100)
    if closes.iloc[-1] > closes.mean() and lows.min() < closes.mean() * 0.95:
        patterns.append("☕ نموذج كوب وعروة محتمل")
    if highs.max() - highs.min() < 0.03 * highs.max():
        patterns.append("🔺 نموذج مثلث صاعد محتمل")
    if lows.max() - lows.min() < 0.03 * lows.max():
        patterns.append("🔻 نموذج مثلث هابط محتمل")
    return patterns

def predict_pattern_with_ai(data, stock_name):
    fig, ax = plt.subplots()
    data_tail = data.tail(60)
    ax.plot(data_tail.index, data_tail['Close'], color='black')
    ax.set_title(f"{stock_name} - Pattern Detection")
    ax.axis('off')
    img_path = f"chart_{stock_name}.png"
    fig.savefig(img_path, bbox_inches='tight')
    plt.close(fig)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model_ai.predict(x)
    decoded = decode_predictions(preds, top=3)[0]
    return [(label, f"{confidence*100:.2f}%") for (_, label, confidence) in decoded]

def analyze_stock(name, symbol):
    data = yf.download(symbol, period="1y", interval="1d")
    data.dropna(inplace=True)
    data["RSI"] = ta.rsi(data["Close"], length=14)
    macd_data = ta.macd(data["Close"])
    data["MACD"] = macd_data.iloc[:, 0]
    data["MACD_signal"] = macd_data.iloc[:, 1]
    data["MFI"] = ta.mfi(data["High"], data["Low"], data["Close"], data["Volume"])
    data["OBV"] = ta.obv(data["Close"], data["Volume"])
    data["SMA_50"] = ta.sma(data["Close"], length=50)
    data["SMA_200"] = ta.sma(data["Close"], length=200)

    latest_close = data["Close"].iloc[-1]
    support = round(data["Close"].rolling(window=20).min().iloc[-1], 2)
    resistance = round(data["Close"].rolling(window=20).max().iloc[-1], 2)

    rsi = data["RSI"].iloc[-1]
    macd = data["MACD"].iloc[-1]
    macd_signal = data["MACD_signal"].iloc[-1]
    mfi = data["MFI"].iloc[-1]
    obv_change = data["OBV"].iloc[-1] - data["OBV"].iloc[-2]
    sma_50 = data["SMA_50"].iloc[-1]
    sma_200 = data["SMA_200"].iloc[-1]

    st.subheader(f"📌 تحليل سهم {name}")
    st.write(f"**📅 التاريخ:** {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    st.write(f"**📉 السعر الحالي:** {latest_close:.2f} ريال")

    st.plotly_chart(go.Figure(data=[
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='شموع'
        )
    ]))

    st.markdown("---")
    st.markdown("### 🧠 المؤشرات الفنية")
    st.write(f"**RSI:** {rsi:.2f}")
    st.write(f"**MACD:** {macd:.2f}, **الإشارة:** {macd_signal:.2f}")
    st.write(f"**MFI:** {mfi:.2f}")
    st.write(f"**OBV:** {'↑' if obv_change > 0 else '↓'}")
    st.write(f"**SMA-50:** {sma_50:.2f} | **SMA-200:** {sma_200:.2f}")
    st.write(f"**الدعم:** {support} | **المقاومة:** {resistance}")

    st.markdown("### ✅ التوصية")
    recommendation = "ترقب – لا توجد إشارة واضحة"
    if rsi < 30 and macd > macd_signal:
        recommendation = "📥 شراء محتمل"
        st.success(recommendation)
    elif rsi > 70 or macd < macd_signal:
        recommendation = "⚠️ احتمال هبوط"
        st.warning(recommendation)
    else:
        st.info(recommendation)

    save_recommendation(name, recommendation)

    st.markdown("### 📐 النماذج الفنية:")
    patterns = detect_chart_patterns(data)
    if patterns:
        for pattern in patterns:
            st.warning(pattern)
    else:
        st.info("لا توجد نماذج حالياً")

    st.markdown("### 🤖 تحليل الذكاء الاصطناعي للنمط:")
    ai_results = predict_pattern_with_ai(data, name)
    for label, conf in ai_results:
        st.write(f"- **{label}**: {conf}")

    st.markdown("### 🔮 توقعات 5 أيام")
    df_prophet = data.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=5)
    forecast = model.predict(future)
    forecast = forecast[["ds", "yhat"]].tail(5).rename(columns={"ds": "التاريخ", "yhat": "السعر المتوقع"})
    st.line_chart(forecast.set_index("التاريخ"))

    if os.path.exists(recommendation_log_file):
        st.markdown("### 🗂️ سجل التوصيات")
        log_df = pd.read_csv(recommendation_log_file)
        st.dataframe(log_df.tail(10))

st.set_page_config(page_title="خبير الأسهم الفني", layout="wide")
st.title("📊 نظام خبير الأسهم الفني السعودي")
st.markdown("نظام شامل لتحليل الأسهم: مؤشرات فنية، نماذج، تنبؤات، ذكاء اصطناعي، إشعارات وتوصيات")
option = st.radio("اختر وضع التشغيل", ("تحليل سهم محدد",))

if option == "تحليل سهم محدد":
    selected_stock = st.selectbox("اختر السهم", list(symbols.keys()))
    analyze_stock(selected_stock, symbols[selected_stock])
