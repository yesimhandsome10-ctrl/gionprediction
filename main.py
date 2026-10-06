# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(page_title="기온 예측기", layout="centered")
st.title("서울 기온 예측기 🌡️")

@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    # 데이터 불러오기
    df = pd.read_csv(url, encoding='utf-8')
    
    # 열 이름 유연하게 매핑 (파일의 실제 열 이름에 '(℃)' 등이 포함되어 있을 수 있음)
    col_map = {}
    for col in df.columns:
        if '날짜' in col: col_map[col] = '날짜'
        elif '평균기온' in col: col_map[col] = '평균기온'
    df = df.rename(columns=col_map)
    
    # 날짜 데이터 변환 및 연도 추출
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['연도'] = df['날짜'].dt.year
    
    # 1. 2025년 이하 데이터만 필터링
    df = df[df['연도'] <= 2025]
    
    # 2. 연도별 관측일수(결측치 제외) 및 평균기온 계산
    yearly_df = df.groupby('연도')['평균기온'].agg(['count', 'mean']).reset_index()
    
    # 3. 관측일이 300일 미만인 해 제외
    yearly_df = yearly_df[yearly_df['count'] >= 300].copy()
    yearly_df = yearly_df.dropna(subset=['mean'])
    
    return yearly_df

# 데이터 준비
yearly_df = load_and_preprocess_data()

# 4. 회귀 분석 및 상관계수 계산 (독립변수: 연도 - 1908)
x = yearly_df['연도'] - 1908
y = yearly_df['mean']

# np.polyfit을 이용해 1차 함수(직선)의 기울기와 절편 구하기
slope, intercept = np.polyfit(x, y, 1)

# 피어슨 상관계수 구하기
correlation = np.corrcoef(x, y)[0, 1]

# 회귀선을 위한 예측값 저장
yearly_df['predicted'] = slope * x + intercept

# 5. UI: 분석 요약 정보 표시
start_year = yearly_df['연도'].min()
end_year = yearly_df['연도'].max()
data_count = len(yearly_df)

st.info(
    f"📊 **분석 요약**\n"
    f"- **사용된 데이터 기간**: {start_year}년 ~ {end_year}년\n"
    f"- **직선을 만든 해의 개수**: {data_count}개 연도\n"
    f"- **상관계수 (연도-기온)**: {correlation:.4f}"
)

# 6. Plotly 그래프 그리기
fig = go.Figure()

# 산점도 (실제 평균기온)
fig.add_trace(go.Scatter(
    x=yearly_df['연도'], 
    y=yearly_df['mean'], 
    mode='markers', 
    name='연도별 평균기온',
    marker=dict(color='royalblue', size=6)
))

# 회귀 직선
fig.add_trace(go.Scatter(
    x=yearly_df['연도'], 
    y=yearly_df['predicted'], 
    mode='lines', 
    name='회귀 직선',
    line=dict(color='red', width=2)
))

fig.update_layout(
    title="서울 연평균기온 변화 및 회귀선",
    xaxis_title="연도",
    yaxis_title="평균기온 (℃)",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# 7. 인터랙티브 기온 예측 (슬라이더)
st.subheader("🔮 연도별 기온 예측")
selected_year = st.slider("예상 기온을 확인할 연도를 선택하세요.", min_value=1900, max_value=2100, value=2026, step=1)

# 예측 공식 적용
predicted_temp = slope * (selected_year - 1908) + intercept

# 예측 결과 크게 표시
st.metric(label=f"{selected_year}년 예상 평균기온", value=f"{predicted_temp:.2f} ℃")
