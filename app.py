# app.py
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="KRX 퀀트 EDA 파이프라인 발표", layout="wide", page_icon="📈"
)

ROOT = Path(__file__).resolve().parent
RAW_PATH = (
    ROOT / "data" / "한국_금융데이터.csv"
    if (ROOT / "data" / "한국_금융데이터.csv").exists()
    else ROOT / "한국_금융데이터.csv"
)
CLEANED_PATH = ROOT / "data" / "processed" / "cleaned.csv"
CHART_DIR = ROOT / "data" / "processed" / "charts"


@st.cache_data
def load_datasets():
    df_raw = (
        pd.read_csv(RAW_PATH, low_memory=False) if RAW_PATH.exists() else None
    )
    df_clean = (
        pd.read_csv(CLEANED_PATH, low_memory=False)
        if CLEANED_PATH.exists()
        else None
    )
    return df_raw, df_clean


df_raw, df_clean = load_datasets()

st.title("📈 KRX 주식 시세 데이터 엔지니어링 & 퀀트 EDA")
st.caption("발표자: 김태겸, 박종원, 서윤하 | 분석기간: 2025-01-02 ~ 2025-12-30 (242거래일)")

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ [Before] 원천 데이터 결함 & 파편화",
    "2️⃣ [Pipeline] main.py 정제 아키텍처",
    "3️⃣ [After] 1:1 복구 및 12대 섹터 체계화",
    "4️⃣ [Insight] 5대 시각화 차트",
])

# ----------------------------------------------------
# [Tab 1] Before: 결측치 실제 사례 및 파편화 문제
# ----------------------------------------------------
with tab1:
    st.header("1. 원천 데이터(Raw Data)의 한계와 실태")

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("총 데이터 행 수", "696,524행", "242거래일 전수")
    col_m2.metric("원천 세부 업종 수", "158개", "극심한 파편화")
    col_m3.metric("원천 결측치 합계", "169,877건", "전체 8% 결측")

    st.subheader("🚨 문제점 1: 우선주·리츠·스팩의 대규모 결측치(NaN)")
    st.markdown(
        "원천 데이터에서는 **우선주, 리츠, 스팩주**의 업종/제품/지역 정보가 완전히 비어 있어(NaN) 통계 집계가 불가능했습니다."
    )

    if df_raw is not None:
        # 결측치가 존재하는 대표 문제 종목들 필터링 샘플 (우선주, 리츠, 스팩)
        problem_mask = df_raw["itmsNm"].str.contains(
            r"(?:우|우선주|리츠|스팩)", na=False
        ) & (df_raw["업종"].isna() | df_raw["주요제품"].isna())
        raw_sample = (
            df_raw[problem_mask][
                [
                    "basDt",
                    "srtnCd",
                    "itmsNm",
                    "mrktCtg",
                    "clpr",
                    "업종",
                    "주요제품",
                    "지역",
                ]
            ]
            .drop_duplicates(subset=["srtnCd"])
            .head(6)
        )

        st.dataframe(raw_sample, use_container_width=True)

    st.subheader("🚨 문제점 2: 158개에 달하는 지나치게 잘게 쪼개진 세부 업종")
    st.markdown(
        "예: `'전자집적회로 제조업'`, `'다이오드 및 기타 반도체 제조업'`, `'통신 및 방송 장비 제조업'` 등으로 파편화되어 있어 **업종별 거시 비교 및 포트폴리오 분석이 불가능**했습니다."
    )

# ----------------------------------------------------
# [Tab 2] Pipeline
# ----------------------------------------------------
with tab2:
    st.header("2. main.py 전처리 및 도메인 복구 파이프라인")
    st.markdown(
        "임의로 행을 삭제하지 않고, **금융 도메인 규칙을 코드로 구현하여 100% 무손실 복구**를 진행했습니다."
    )

    st.code(
        """
    [4단계 정제 파이프라인]
    1. 본주(Parent Company) 역추적:
       - '현대차3우(전환)' ➔ 본주 '현대차'의 업종('자동차/운송장비')으로 자동 복구
    2. 스팩(SPAC) / 리츠(REITs) 금융 도메인 표준화:
       - 스팩주 ➔ '금융 및 보험업' / '기업인수 및 합병' / '서울특별시'
       - 리츠주 ➔ '부동산업' / '부동산 투자 및 임대' / '서울특별시'
    3. 잔여 결측치 ➔ '기타(미분류)' 명시적 범주화 (NaN 제거)
    4. 158개 세부 업종 ➔ 12대 핵심 퀀트 대분류 섹터(sector_macro)로 그룹화 (압축률 92%)
    """,
        language="python",
    )

# ----------------------------------------------------
# [Tab 3] After: 1:1 완벽 복구 및 12대 섹터
# ----------------------------------------------------
with tab3:
    st.header("3. 정제 완료 데이터 (Cleaned Dataset)")

    c1, c2, c3 = st.columns(3)
    c1.metric("정제 후 결측률", "0.00%", "-169,877건 완벽 복구")
    c2.metric("업종 분류 체계", "12대 퀀트 섹터", "158개 ➔ 12개 체계화")
    c3.metric("데이터 무결성", "696,524행", "유실 0건 보존")

    st.subheader("✨ 1:1 매칭 복구 결과 (비어있던 우선주·리츠·스팩이 채워진 모습)")
    if df_clean is not None:
        # 1탭에서 보여준 문제 종목들의 정제 후 모습 조회
        target_codes = [
            "005387",
            "005935",
            "334890",
            "002795",
            "0093G0",
            "000300",
        ]
        clean_sample = (
            df_clean[
                df_clean["srtnCd"].isin(target_codes)
                | df_clean["itmsNm"].str.contains(r"(?:우|리츠|스팩)")
            ][
                [
                    "basDt",
                    "srtnCd",
                    "itmsNm",
                    "mrktCtg",
                    "is_preferred",
                    "업종",
                    "sector_macro",
                ]
            ]
            .drop_duplicates(subset=["srtnCd"])
            .head(6)
        )

        st.dataframe(clean_sample, use_container_width=True)

# ----------------------------------------------------
# [Tab 4] Charts
# ----------------------------------------------------
with tab4:
    st.header("4. 정제된 데이터를 통한 5대 시각화 및 인사이트")

    col1, col2 = st.columns(2)
    with col1:
        st.image(
            str(CHART_DIR / "chart1_market_and_top_sectors.png"),
            caption="차트 1. 시장별 비중 & 12대 섹터 거래대금",
        )
        st.image(
            str(CHART_DIR / "chart3_market_boxplots.png"),
            caption="차트 3. 시장별 등락률 & 거래대금 Box Plot (가독성 개선)",
        )
    with col2:
        st.image(
            str(CHART_DIR / "chart4_market_timeseries_ma20.png"),
            caption="차트 4. 2025년 시장 거래대금 추이 & 20일 이평선",
        )
        st.image(
            str(CHART_DIR / "chart5_sector_and_intraday_momentum.png"),
            caption="차트 5. 12대 섹터별 양봉 vs 음봉 비율 (장중 음봉 우세 입증)",
        )