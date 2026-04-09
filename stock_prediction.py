import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from datetime import timedelta
import os

# 1. 페이지 설정
st.set_page_config(
    page_title="MSFT 주가 예측 및 기술적 분석",
    page_icon="📈",
    layout="wide"
)

# 프리미엄 스타일을 위한 커스텀 CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #161b22;
        border-radius: 5px;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. 자산 로드 (캐싱)
@st.cache_resource
def load_assets():
    model_path = 'model/best_gru_model.keras'
    scaler_path = 'model/stock_scaler.pkl'
    data_path = 'model/stock_data_refined.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(data_path):
        st.error("모델 또는 데이터 파일을 찾을 수 없습니다.")
        st.stop()
            
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    data = joblib.load(data_path)
    return model, scaler, data

try:
    model, scaler, df = load_assets()
except Exception as e:
    st.error(f"데이터 로드 오류: {e}")
    st.stop()

# 날짜 컬럼 및 기술적 지표 계산
df['Date'] = pd.to_datetime(df['Date'])
target_col = 'MSFT'
df['MA20'] = df[target_col].rolling(window=20).mean()
df['MA50'] = df[target_col].rolling(window=50).mean()

# 3. 데이터 전처리
prices = df[target_col].values.reshape(-1, 1)
scaled_prices = scaler.transform(prices)

seq_length = 60
X = []
for i in range(seq_length, len(scaled_prices)):
    X.append(scaled_prices[i-seq_length:i])

X = np.array(X) 

# 4. 모델 추론
with st.spinner("분석 엔진 가동 중..."):
    predictions_scaled = model.predict(X, verbose=0)
    predictions = scaler.inverse_transform(predictions_scaled)

    actual_prices = prices[seq_length:]
    dates = df['Date'].values[seq_length:]
    
    # 내일 날짜 예측
    last_window = scaled_prices[-seq_length:].reshape(1, seq_length, 1)
    next_day_pred_scaled = model.predict(last_window, verbose=0)
    next_day_pred = scaler.inverse_transform(next_day_pred_scaled)[0][0]
    
    last_date = df['Date'].iloc[-1]
    next_date = last_date + timedelta(days=1)
    if last_date.weekday() >= 4:
        next_date = last_date + timedelta(days=(7 - last_date.weekday()))

# 5. 사이드바 구성
st.sidebar.header("🛠️ 대시보드 옵션")
show_ma = st.sidebar.toggle("이동 평균선(MA) 표시", value=True)
selected_date = st.sidebar.date_input(
    "조회 날짜 선택",
    value=pd.to_datetime(dates[-1]),
    min_value=pd.to_datetime(dates[0]),
    max_value=pd.to_datetime(dates[-1])
)

# 데이터 인덱싱
target_dt = pd.to_datetime(selected_date)
idx = np.argmin(np.abs(pd.to_datetime(dates) - target_dt))
view_date_str = pd.to_datetime(dates[idx]).strftime('%Y-%m-%d')
actual_val = actual_prices[idx][0]
pred_val = predictions[idx][0]

st.sidebar.divider()
st.sidebar.subheader(f"📅 {view_date_str} 분석")
st.sidebar.write(f"**실제 종가:** ${actual_val:.2f}")
st.sidebar.write(f"**GRU 예측:** ${pred_val:.2f}")

# 2. 기존의 '변동성'을 '전일 대비 등락'으로 이름 변경
st.sidebar.write(f"**전일 대비 등락:** {((actual_val - actual_prices[idx-1][0]) if idx > 0 else 0):+.2f} USD")

# VIX 변동성 로직
vix_values = df['VIX'].values[seq_length:] 
selected_vix = vix_values[idx]
vix_status = "🔴 주의" if selected_vix >= 20 else "🟢 안정"
st.sidebar.write(f"**시장 변동성(VIX):** {selected_vix:.2f} ({vix_status})")

# 6. 메인 화면 구성
st.title("📈 MSFT 스마트 주가 분석 시스템")
st.markdown(f"마이크로소프트 전용 GRU 예측 모델과 기술적 지표를 결합한 대시보드입니다. (업데이트: **{last_date.strftime('%Y-%m-%d')}**)")

# 주요 지표
c1, c2, c3 = st.columns(3)
cur_price = prices[-1][0]
diff = next_day_pred - cur_price
pct = (diff / cur_price) * 100

c1.metric("현재 주가", f"${cur_price:.2f}")
c2.metric(f"{next_date.strftime('%Y-%m-%d')} 예측", f"${next_day_pred:.2f}", f"{diff:+.2f} ({pct:+.2f}%)")
c3.metric("시장 상태", "안정성 높음" if df['VIX'].iloc[-1] < 20 else "변동성 주의")

# 7. 시각화 섹션
t1, t2 = st.tabs(["🚀 가격 예측 및 이평선", "💰 수익률 시뮬레이션"])

with t1:
    st.subheader("실제 주가 및 예측 데이터 시연")
    fig = go.Figure()
    
    # 실제 주가
    fig.add_trace(go.Scatter(x=dates, y=actual_prices.flatten(), name="실제 종가", line=dict(color='#00d1ff', width=2)))
    # GRU 예측
    fig.add_trace(go.Scatter(x=dates, y=predictions.flatten(), name="GRU 예측가", line=dict(color='#ff4b4b', width=2, dash='dash')))
    
    if show_ma:
        ma20_view = df['MA20'].values[seq_length:]
        ma50_view = df['MA50'].values[seq_length:]
        fig.add_trace(go.Scatter(x=dates, y=ma20_view, name="20일 이평선", line=dict(color='#ffcc00', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=dates, y=ma50_view, name="50일 이평선", line=dict(color='#00ffcc', width=1, dash='dot')))
        
    fig.add_vline(x=selected_date, line_width=1, line_dash="solid", line_color="white")
    
    fig.update_layout(template="plotly_dark", height=600, xaxis_title="날짜", yaxis_title="주가 (USD)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with t2:
    st.subheader("누적 수익률 분석 (종가 기준)")
    st.markdown("선택한 투자 시작일로부터의 주가 상승률을 기반으로 한 수익률 변화입니다.")
    
    # 1. 투자 시작일 선택 UI 생성
    selected_start_date = st.selectbox("투자 시작일 선택", dates)
    
    # 2. 선택한 날짜의 인덱스 찾기
    start_idx = list(dates).index(selected_start_date)
    
    # 💡 3. [추가된 부분] 선택한 날짜의 종가(시작 가격) 추출 및 화면에 표시
    start_price = actual_prices[start_idx][0]
    date_str = pd.to_datetime(selected_start_date).strftime('%Y-%m-%d')
    st.info(f"📌 선택하신 **{date_str}**의 기준 종가는 **${start_price:.2f}** 입니다.")
    
    # 4. 선택한 날짜 이후의 데이터만 슬라이싱
    filtered_dates = dates[start_idx:]
    filtered_prices = actual_prices[start_idx:]
    
    # 5. 수익률 계산 (선택한 날짜의 종가를 '0번째'로 두고 계산)
    investment = 1000
    cum_returns = (filtered_prices / filtered_prices[0]) - 1
    asset_value = (1 + cum_returns) * investment
    
    # 6. 차트 그리기
    fig_rev = go.Figure()
    
    # 누적 수익률 선 추가
    fig_rev.add_trace(go.Scatter(
        x=filtered_dates, 
        y=(cum_returns.flatten() * 100), # % 단위로 변환
        name="누적 수익률 (%)", 
        fill='tozeroy', 
        line=dict(color='#00ff88', width=2)
    ))
    
    fig_rev.update_layout(
        template="plotly_dark", 
        height=500, 
        xaxis_title="날짜", 
        yaxis_title="수익률 (%)", 
        hovermode="x unified"
    )
    st.plotly_chart(fig_rev, use_container_width=True)
    
    # 7. 최종 결과 요약 텍스트 업데이트
    final_ret = cum_returns[-1][0] * 100
    final_asset = asset_value[-1][0]
    st.success(f"해당 기간 동안의 총 수익률: **{final_ret:+.2f}% (초기 $1,000 투자 시 현재 가치: ${final_asset:.2f}**)")

# 8. 데이터 가시화
with st.expander("📊 4대 기술주 원본 데이터 및 상세 지표 "):
    st.dataframe(df.tail(30), use_container_width=True)
    st.caption("표시된 데이터는 최근 4대 기술주의 6주간 지표입니다.")

st.markdown("---")
st.caption("면책 조항: 본 모델의 예측은 참고용이며 금융 투자 결과에 대한 책임을 지지 않습니다.")
