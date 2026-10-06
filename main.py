# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 페이지 설정
st.set_page_config(page_title="기온 예측 모델 평가", layout="wide")
st.title("서울 기온 선형회귀 모델 평가 및 비교 🌡️")

@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding='utf-8')
    
    # 열 이름 유연하게 매핑
    col_map = {}
    for col in df.columns:
        if '날짜' in col: col_map[col] = '날짜'
        elif '평균기온' in col: col_map[col] = '평균기온'
    df = df.rename(columns=col_map)
    
    # 날짜 데이터 변환 및 연도 추출
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['연도'] = df['날짜'].dt.year
    
    # 2025년 이하 데이터만 필터링
    df = df[df['연도'] <= 2025]
    
    # 연도별 관측일수 및 평균기온 계산
    yearly_df = df.groupby('연도')['평균기온'].agg(['count', 'mean']).reset_index()
    
    # 관측일이 300일 미만인 해 및 결측치 제외
    yearly_df = yearly_df[yearly_df['count'] >= 300].copy()
    yearly_df = yearly_df.dropna(subset=['mean'])
    
    return yearly_df

yearly_df = load_and_preprocess_data()

# 데이터 분할 기준
# 전체 데이터: 1908 ~ 2025
# 테스트 데이터: 최근 20년 (2006 ~ 2025)
# 훈련 데이터 1: 최근 50년 (1956 ~ 2005)
# 훈련 데이터 2: 최근 100년 (1906 ~ 2005) - 실제 데이터는 1908년부터 존재

test_df = yearly_df[(yearly_df['연도'] >= 2006) & (yearly_df['연도'] <= 2025)]
train_50_df = yearly_df[(yearly_df['연도'] >= 1956) & (yearly_df['연도'] <= 2005)]
train_100_df = yearly_df[(yearly_df['연도'] >= 1906) & (yearly_df['연도'] <= 2005)]

# 평가 함수 정의
def train_and_evaluate(train_data, test_data):
    # scikit-learn은 2차원 배열의 X(독립변수)를 요구함
    X_train = train_data[['연도']]
    y_train = train_data['mean']
    
    X_test = test_data[['연도']]
    y_test = test_data['mean']
    
    # 모델 학습
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 예측 및 평가
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    slope = model.coef_[0] # 기울기 (1년당 기온 변화량)
    
    return model, mae, mse, r2, slope

# 1. 전체 데이터 모델 평가 (Train = All, Test = All)
model_all, mae_all, mse_all, r2_all, slope_all = train_and_evaluate(yearly_df, yearly_df)

# 2. 최근 50년 학습 모델 평가 (Train = 50년, Test = 20년)
model_50, mae_50, mse_50, r2_50, slope_50 = train_and_evaluate(train_50_df, test_df)

# 3. 최근 100년 학습 모델 평가 (Train = 100년, Test = 20년)
model_100, mae_100, mse_100, r2_100, slope_100 = train_and_evaluate(train_100_df, test_df)


# ---------------- UI 구성 ---------------- #
st.header("1. 전체 데이터 회귀선 평가 (1908~2025)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("기울기 (년당 상승폭)", f"{slope_all:.4f} ℃")
col2.metric("MAE (평균 절대 오차)", f"{mae_all:.4f}")
col3.metric("MSE (평균 제곱 오차)", f"{mse_all:.4f}")
col4.metric("R² (결정계수)", f"{r2_all:.4f}")

st.divider()

st.header("2. 학습 기간에 따른 예측 성능 비교")
st.markdown("과거 데이터를 학습하여 **최근 20년(2006~2025)**의 기온을 얼마나 잘 예측하는지 테스트합니다.")

# 비교 테이블 생성
comparison_data = {
    "학습 데이터 기간": ["최근 50년 (1956~2005)", "최근 100년 (1908~2005)"],
    "기울기 (연간 기온 상승폭)": [f"{slope_50:.4f} ℃/년", f"{slope_100:.4f} ℃/년"],
    "MAE (평균 절대 오차)": [f"{mae_50:.4f}", f"{mae_100:.4f}"],
    "MSE (평균 제곱 오차)": [f"{mse_50:.4f}", f"{mse_100:.4f}"],
    "R² (결정계수)": [f"{r2_50:.4f}", f"{r2_100:.4f}"]
}
st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

with st.expander("📊 평가 지표 해석 보기"):
    st.markdown("""
    * **기울기**: 최근 50년만 학습했을 때 가팔라지는지(더 빠르게 온난화가 진행되는지) 확인할 수 있습니다.
    * **MAE (오차)**: 실제 값과 예측 값 차이의 평균입니다. 작을수록 예측이 정확합니다.
    * **MSE (오차)**: 오차를 제곱하여 평균 낸 값으로, 큰 오차에 패널티를 줍니다. 작을수록 좋습니다.
    * **R² (결정계수)**: 1에 가까울수록 회귀선이 데이터를 잘 설명한다는 뜻입니다. (음수가 나올 경우, 회귀선이 단순 평균값으로 예측하는 것보다도 정확도가 떨어진다는 의미입니다.)
    """)

# 그래프 시각화
st.header("3. 모델별 회귀 직선 시각화")

fig = go.Figure()

# 실제 데이터 산점도 (학습/테스트 색상 구분)
train_all_df = yearly_df[yearly_df['연도'] <= 2005]
fig.add_trace(go.Scatter(
    x=train_all_df['연도'], y=train_all_df['mean'], mode='markers', 
    name='과거 데이터 (~2005)', marker=dict(color='lightgray', size=6)
))

fig.add_trace(go.Scatter(
    x=test_df['연도'], y=test_df['mean'], mode='markers', 
    name='테스트 데이터 (2006~2025)', marker=dict(color='orange', size=8)
))

# 예측 직선 추가를 위한 연도 범위 (그래프 표시용)
x_plot = pd.DataFrame({'연도': np.arange(1900, 2030)})

# 전체 데이터 기준 회귀선
fig.add_trace(go.Scatter(
    x=x_plot['연도'], y=model_all.predict(x_plot), mode='lines', 
    name='전체 데이터 기준 (1908~2025)', line=dict(color='green', width=2, dash='dot')
))

# 100년 학습 기준 회귀선
fig.add_trace(go.Scatter(
    x=x_plot['연도'], y=model_100.predict(x_plot), mode='lines', 
    name='최근 100년 학습 (1908~2005)', line=dict(color='blue', width=2)
))

# 50년 학습 기준 회귀선
fig.add_trace(go.Scatter(
    x=x_plot['연도'], y=model_50.predict(x_plot), mode='lines', 
    name='최근 50년 학습 (1956~2005)', line=dict(color='red', width=2)
))

fig.update_layout(
    title="서울 연평균기온 변화 및 학습 기간별 회귀 직선 비교",
    xaxis_title="연도",
    yaxis_title="평균기온 (℃)",
    template="plotly_white",
    hovermode="x unified"
)

# 이전 로그의 경고를 해결하기 위해 use_container_width=True 대신 width='stretch' 사용 권장됨
# Streamlit 버전에 따라 동작이 다를 수 있어 st.plotly_chart 매개변수로 안전하게 처리합니다.
st.plotly_chart(fig, use_container_width=True)
