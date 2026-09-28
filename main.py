import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress

# 페이지 기본 설정
st.set_page_config(page_title="서울 기온 예측기", layout="centered")
st.title("🌡️ 서울 기온 예측기")

@st.cache_data
def load_and_preprocess_data():
    # 데이터 불러오기
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    
    # 지정된 열 이름으로 데이터 읽기
    df = pd.read_csv(url, encoding='utf-8', names=['날짜', '지점', '평균기온', '최저기온', '최고기온'], header=0)
    
    # 날짜 데이터 처리 및 연도 추출
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['연도'] = df['날짜'].dt.year
    
    # 연도별 평균기온과 관측일수(데이터 개수) 계산
    yearly_stats = df.groupby('연도')['평균기온'].agg(['mean', 'count']).reset_index()
    yearly_stats.rename(columns={'mean': '연평균기온', 'count': '관측일수'}, inplace=True)
    
    # 조건 필터링: 2025년 이하, 관측일수 300일 이상
    filtered_data = yearly_stats[(yearly_stats['연도'] <= 2025) & (yearly_stats['관측일수'] >= 300)]
    
    return filtered_data

# 데이터 로드
data = load_and_preprocess_data()

# 회귀 분석 (독립변수: 1908년부터 지난 연수)
X = data['연도'] - 1908
Y = data['연평균기온']

slope, intercept, r_value, p_value, std_err = linregress(X, Y)

# 화면에 기본 정보 출력
st.subheader("📊 데이터 요약 및 상관관계")
st.write(f"- **분석에 사용된 해의 개수**: {len(data)}년")
st.write(f"- **데이터 시작 연도**: {data['연도'].min()}년")
st.write(f"- **데이터 끝 연도**: {data['연도'].max()}년")
st.write(f"- **상관계수 (r)**: {r_value:.4f}")

# Plotly 그래프 생성
fig = go.Figure()

# 실제 데이터 산점도 추가
fig.add_trace(go.Scatter(
    x=data['연도'], 
    y=data['연평균기온'], 
    mode='markers', 
    name='연평균기온 (관측값)',
    marker=dict(color='royalblue')
))

# 회귀 직선 계산 및 추가 (가로축은 연도 그대로 표시)
line_x = np.array([data['연도'].min(), data['연도'].max()])
line_y = slope * (line_x - 1908) + intercept

fig.add_trace(go.Scatter(
    x=line_x, 
    y=line_y, 
    mode='lines', 
    name='추세선 (회귀 직선)',
    line=dict(color='red', width=2)
))

fig.update_layout(
    title="연도별 서울 평균기온 변화 및 추세선",
    xaxis_title="연도",
    yaxis_title="평균기온 (℃)",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# 기온 예측 슬라이더 및 결과 표시
st.subheader("🔮 특정 연도의 예상 기온 알아보기")

selected_year = st.slider("연도를 선택하세요:", min_value=1900, max_value=2100, value=2025)

# 회귀 식을 이용한 기온 예측 (y = ax + b, x는 1908년부터 지난 연수)
predicted_temp = slope * (selected_year - 1908) + intercept

# 결과를 크게 표시
st.metric(label=f"{selected_year}년 예상 평균기온", value=f"{predicted_temp:.2f} ℃")QQ  
