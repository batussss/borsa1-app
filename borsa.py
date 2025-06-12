import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import snscrape.modules.twitter as sntwitter
import matplotlib.pyplot as plt
from prophet import Prophet
import requests

# ---------------- Teknik Analiz Fonksiyonları ----------------
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(data):
    ema12 = data.ewm(span=12, adjust=False).mean()
    ema26 = data.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def compute_bollinger_bands(data, window=20):
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper_band = sma + (2 * std)
    lower_band = sma - (2 * std)
    return upper_band, lower_band

def generate_signals(df):
    close = df['Close']
    last_price = close.iloc[-1]
    rsi = compute_rsi(close).iloc[-1]
    macd, macd_signal = compute_macd(close)
    macd_diff = macd.iloc[-1] - macd_signal.iloc[-1]
    ma20 = close.rolling(window=20).mean().iloc[-1]
    upper_band, lower_band = compute_bollinger_bands(close)
    bb_upper = upper_band.iloc[-1]
    bb_lower = lower_band.iloc[-1]

    if rsi < 30 and last_price < ma20 and macd_diff > 0 and last_price < bb_lower:
        return "AL"
    elif rsi > 70 and last_price > ma20 and macd_diff < 0 and last_price > bb_upper:
        return "SAT"
    else:
        return "BEKLE"

# ---------------- Twitter Verisi Çekme ----------------
def fetch_tweets(query, max_tweets=10):
    tweets_list = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= max_tweets:
            break
        tweets_list.append({
            'date': tweet.date,
            'content': tweet.content,
            'username': tweet.user.username,
            'url': tweet.url
        })
    return pd.DataFrame(tweets_list)

# ---------------- Backtest Fonksiyonu ----------------
def backtest_multiple_days(ticker, days=5):
    results = []
    today = datetime.now().date()
    day_count = 0
    checked_days = 0

    while checked_days < days:
        day = today - timedelta(days=day_count+1)
        day_str = day.strftime('%Y-%m-%d')
        next_day_str = (day + timedelta(days=1)).strftime('%Y-%m-%d')
        day_data = ticker.history(start=day_str, end=next_day_str, interval="15m")
        day_count += 1

        if day_data.empty:
            continue

        signal = generate_signals(day_data)
        open_price = day_data['Close'].iloc[0]
        close_price = day_data['Close'].iloc[-1]
        delta_pct = (close_price - open_price) / open_price * 100

        if (signal == "AL" and delta_pct > 0) or (signal == "SAT" and delta_pct < 0):
            correct = True
        elif signal == "BEKLE":
            correct = None
        else:
            correct = False

        results.append({
            'date': day_str,
            'signal': signal,
            'delta_pct': delta_pct,
            'correct': correct
        })

        checked_days += 1

    return pd.DataFrame(results)

# ---------------- Prophet Tahmin ----------------
def prophet_forecast(data, days=3):
    df = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast

# ---------------- Telegram Bot Fonksiyonları ----------------
def send_telegram_message(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    response = requests.post(url, data=payload)
    return response.status_code == 200

# ---------------- Streamlit Arayüz ----------------
st.title("📈 Gelişmiş Anlık Borsa Paneli")

st.sidebar.title("🔎 Hisse Seçimi & Telegram Ayarları")
symbol_input = st.sidebar.text_input("Hisse Sembolü (örn: ASELS.IS):", "ASELS.IS").upper()

st.sidebar.markdown("---")
st.sidebar.header("📡 Telegram Bot Ayarları")
bot_token = st.sidebar.text_input("Bot Token (örn: 123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11)")
chat_id = st.sidebar.text_input("Chat ID (örn: 123456789)")

ticker = yf.Ticker(symbol_input)
data = ticker.history(period="60d", interval="1d")  # Prophet için
intraday_data = ticker.history(period="5d", interval="15m")  # Teknik analiz

tabs = st.tabs(["📈 Teknik Analiz", "💬 Yorumlar", "🤖 AI Tahmin", "📡 Telegram & Bot"])

# --- Tab 1: Teknik Analiz ---
with tabs[0]:
    st.header(f"{symbol_input} - Teknik Analiz")

    if intraday_data.empty:
        st.error("Veri alınamadı veya sembol geçersiz.")
    else:
        st.subheader("Son 5 Gün 15 Dakikalık Kapanış Fiyatları")
        st.line_chart(intraday_data['Close'])

        signal = generate_signals(intraday_data)
        rsi_value = compute_rsi(intraday_data['Close']).iloc[-1]
        macd_val, macd_signal = compute_macd(intraday_data['Close'])
        macd_diff = macd_val.iloc[-1] - macd_signal.iloc[-1]
        last_price = intraday_data['Close'].iloc[-1]

        st.metric("Son Fiyat", f"{last_price:.2f} TL")
        st.metric("RSI (14)", f"{rsi_value:.2f}")
        st.metric("MACD Diff", f"{macd_diff:.4f}")
        st.metric("Sinyal", signal)

        if signal == "AL":
            st.success("Teknik göstergeler AL sinyali veriyor.")
        elif signal == "SAT":
            st.error("Teknik göstergeler SAT sinyali veriyor.")
        else:
            st.info("Sinyaller kararsız.")

        st.markdown("---")
        st.markdown("### 🧪 Son 5 Günlük Backtest Performansı")
        backtest_df = backtest_multiple_days(ticker, days=5)

        if not backtest_df.empty:
            st.dataframe(backtest_df)

            valid_results = backtest_df[backtest_df['correct'].notnull()]
            if not valid_results.empty:
                accuracy = valid_results['correct'].mean() * 100
                st.metric("Backtest Doğruluk Oranı", f"%{accuracy:.2f}")

                fig, ax = plt.subplots(figsize=(8,4))
                colors = valid_results['correct'].map({True:'green', False:'red'})
                ax.bar(valid_results['date'], valid_results['delta_pct'], color=colors)
                ax.axhline(0, color='black', linewidth=0.8)
                ax.set_ylabel("Günlük Fiyat Değişimi (%)")
                ax.set_title("Backtest Günlük Performansı ve Doğruluk")
                for i, val in enumerate(valid_results['delta_pct']):
                    ax.text(i, val + (0.5 if val > 0 else -2), f"{val:.2f}%", ha='center', color='black')
                st.pyplot(fig)
            else:
                st.info("Backtest sinyalleri kararsız.")
        else:
            st.warning("Backtest verisi bulunamadı.")

# --- Tab 2: Sosyal Medya Yorumları ---
with tabs[1]:
    st.header(f"{symbol_input} - Sosyal Medya Yorumları")

    tradingview_tab, twitter_tab = st.tabs(["TradingView", "Twitter"])

    with tradingview_tab:
        st.markdown(f"### TradingView Yorumları")
        tv_url = f"https://tr.tradingview.com/symbols/{symbol_input.split('.')[0]}/"
        st.markdown(f'<iframe src="{tv_url}" width="100%" height="600px" frameborder="0"></iframe>', unsafe_allow_html=True)
        st.info("TradingView sayfası yüklendi.")

    with twitter_tab:
        st.markdown(f"### Twitter Son 10 Tweet")
        query = f"${symbol_input.split('.')[0]} lang:tr"
        tweets_df = fetch_tweets(query, max_tweets=10)
        if tweets_df.empty:
            st.info("Tweet bulunamadı.")
        else:
            for _, row in tweets_df.iterrows():
                st.markdown(f"**@{row['username']}** - {row['date'].strftime('%Y-%m-%d %H:%M')}")
                st.write(row['content'])
                st.markdown(f"[Tweet Linki]({row['url']})")
                st.markdown("---")

# --- Tab 3: AI Tahmin (Prophet) ---
with tabs[2]:
    st.header(f"{symbol_input} - Fiyat Tahmini (Prophet)")

    if data.empty or len(data) < 30:
        st.warning("Yeterli veri yok veya sembol hatalı.")
    else:
        forecast = prophet_forecast(data, days=3)
        
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(data.index, data['Close'], label='Gerçek Fiyat')
        ax.plot(forecast['ds'], forecast['yhat'], label='Tahmin', linestyle='--')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2, label='Tahmin Aralığı')
        ax.set_title(f"{symbol_input} - 3 Günlük Fiyat Tahmini")
        ax.set_xlabel("Tarih")
        ax.set_ylabel("Fiyat")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

# --- Tab 4: Telegram & Bot ---
with tabs[3]:
    st.header("📡 Telegram & Bot")

    st.write("Telegram botunuz ile mesaj gönderin.")

    message = st.text_area("Gönderilecek Mesaj", "")

    if st.button("Mesajı Gönder"):
        if bot_token.strip() == "" or chat_id.strip() == "":
            st.error("Lütfen Bot Token ve Chat ID giriniz.")
        elif message.strip() == "":
            st.error("Lütfen mesaj giriniz.")
        else:
            success = send_telegram_message(bot_token, chat_id, message)
            if success:
                st.success("Mesaj Telegram'a başarıyla gönderildi!")
            else:
                st.error("Mesaj gönderilirken hata oluştu. Token, Chat ID veya bağlantıyı kontrol edin.")
