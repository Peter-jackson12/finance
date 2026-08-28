from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ----------------------------------------------------
# 1. 환경 설정 및 폰트
# ----------------------------------------------------
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
CLEANED_PATH = ROOT / "data" / "processed" / "cleaned.csv"
CHART_DIR = ROOT / "data" / "processed" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------
# 2. 정밀화된 12대 섹터 매핑 룰 (기타/서비스 과밀 해소)
# ----------------------------------------------------
def map_macro_sector_refined(industry_name: str) -> str:
    ind = str(industry_name).strip()

    if any(
        k in ind
        for k in [
            "반도체",
            "전자",
            "집적회로",
            "컴퓨터",
            "디스플레이",
            "통신장비",
            "방송장비",
            "광학",
            "전기전자",
            "IT",
        ]
    ):
        return "IT/반도체"
    elif any(
        k in ind
        for k in [
            "의약",
            "바이오",
            "제약",
            "의료",
            "생물",
            "병원",
            "헬스케어",
            "치료",
            "보건",
        ]
    ):
        return "바이오/헬스케어"
    elif any(
        k in ind
        for k in [
            "소프트웨어",
            "게임",
            "프로그래밍",
            "포털",
            "정보서비스",
            "엔터",
            "방송",
            "영화",
            "음원",
            "출판",
            "광고",
            "콘텐츠",
        ]
    ):
        return "미디어/컨텐츠/게임"
    elif any(
        k in ind
        for k in [
            "전지",
            "축전지",
            "화학",
            "석유",
            "정유",
            "고무",
            "플라스틱",
            "에너지",
            "도료",
            "비료",
        ]
    ):
        return "2차전지/화학"
    elif any(
        k in ind
        for k in [
            "자동차",
            "트레일러",
            "부품",
            "조선",
            "선박",
            "항공",
            "철도",
            "운송장비",
            "모터",
        ]
    ):
        return "자동차/운송장비"
    elif any(
        k in ind
        for k in [
            "기계",
            "로봇",
            "장비",
            "공작",
            "엔진",
            "펌프",
            "밸브",
            "전기",
            "전력",
            "가스",
            "유틸리티",
            "발전",
        ]
    ):
        return "기계/유틸리티"
    elif any(
        k in ind
        for k in [
            "금융",
            "지주",
            "은행",
            "증권",
            "보험",
            "투자",
            "캐피탈",
            "카드",
            "자산운용",
            "스팩",
            "SPAC",
        ]
    ):
        return "금융/지주"
    elif any(
        k in ind
        for k in [
            "음식료",
            "식품",
            "음료",
            "제과",
            "축산",
            "수산",
            "의복",
            "패션",
            "섬유",
            "신발",
            "화장품",
            "뷰티",
            "생활용품",
            "가구",
        ]
    ):
        return "소비재/식음료/패션"
    elif any(
        k in ind
        for k in [
            "도매",
            "소매",
            "유통",
            "상사",
            "무역",
            "중개",
            "백화점",
            "운수",
            "창고",
            "택배",
            "물류",
            "해운",
        ]
    ):
        return "유통/물류/상사"
    elif any(
        k in ind
        for k in [
            "철강",
            "제철",
            "제강",
            "금속",
            "비금속",
            "광물",
            "시멘트",
            "유리",
            "세라믹",
            "제지",
            "목재",
        ]
    ):
        return "철강/소재"
    elif any(
        k in ind
        for k in [
            "건설",
            "토목",
            "건축",
            "플랜트",
            "엔지니어링",
            "부동산",
            "리츠",
        ]
    ):
        return "건설/부동산"
    else:
        return "기타/서비스"


def generate_all_report_charts():
    print(f"📊 [1/5] 데이터 로딩 및 섹터 재분류 중: {CLEANED_PATH.name}")
    df = pd.read_csv(CLEANED_PATH, encoding="utf-8-sig", low_memory=False)
    df["basDt"] = pd.to_datetime(df["basDt"].astype(str))

    # 정밀화된 섹터 적용
    df["sector_macro"] = df["업종"].apply(map_macro_sector_refined)

    # ----------------------------------------------------
    # [차트 1] 시장별 거래 건수 & 재분류된 12대 섹터별 거래대금
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    market_order = ["KOSPI", "KOSDAQ", "KONEX"]
    market_counts = df["mrktCtg"].value_counts().reindex(market_order)

    axes[0].bar(
        market_counts.index,
        market_counts.values,
        color=["#4C72B0", "#55A868", "#C44E52"],
        edgecolor="black",
        width=0.6,
    )
    axes[0].set_title(
        "시장별 거래 레코드 수 (Market Distribution)",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].set_ylabel("데이터 건수 (행)")
    for i, v in enumerate(market_counts.values):
        axes[0].text(
            i,
            v + 6000,
            f"{v:,}건\n({v/len(df)*100:.1f}%)",
            ha="center",
            fontsize=10,
        )

    sector_tr = (
        df.groupby("sector_macro")["trPrc"]
        .sum()
        .sort_values(ascending=True)
        / 1e12
    )
    # height -> width=0.6 으로 수정 완료
    sector_tr.plot(
        kind="barh", ax=axes[1], color="#008080", edgecolor="black", width=0.6
    )
    axes[1].set_title(
        "12대 대분류 섹터별 연간 총 거래대금 (단위: 조 원)",
        fontsize=13,
        fontweight="bold",
    )
    axes[1].set_xlabel("총 거래대금 (조 원)")
    plt.tight_layout()
    plt.savefig(
        CHART_DIR / "chart1_market_and_top_sectors.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ----------------------------------------------------
    # [차트 2] 일별 등락률 분포 & 시총-거래대금 산점도
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.histplot(
        df["fltRt"].clip(-15, 15),
        bins=50,
        kde=True,
        ax=axes[0],
        color="royalblue",
    )
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5, label="0% 기준선")
    axes[0].set_title(
        "일별 등락률(fltRt) 분포 (-15% ~ +15% 클리핑)",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].set_xlabel("등락률 (%)")
    axes[0].legend()

    sample_df = df.sample(min(10000, len(df)), random_state=42)
    sns.scatterplot(
        data=sample_df,
        x="mrktTotAmt",
        y="trPrc",
        hue="mrktCtg",
        hue_order=market_order,
        alpha=0.5,
        ax=axes[1],
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title(
        "시가총액 vs 거래대금 (Log-Log Scale, 멱법칙 확인)",
        fontsize=13,
        fontweight="bold",
    )
    axes[1].set_xlabel("시가총액 (Log Scale, 원)")
    axes[1].set_ylabel("거래대금 (Log Scale, 원)")
    plt.tight_layout()
    plt.savefig(
        CHART_DIR / "chart2_fltrt_dist_and_scatter.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ----------------------------------------------------
    # [차트 3] 가독성 극대화 박스플롯 (이상치 점 숨김 & 순서 통일)
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1) 시장별 등락률 (showfliers=False 적용)
    sns.boxplot(
        data=df,
        x="mrktCtg",
        y="fltRt",
        order=market_order,
        showfliers=False,
        palette=["#4C72B0", "#55A868", "#C44E52"],
        ax=axes[0],
        width=0.5,
    )
    axes[0].axhline(0, color="red", linestyle=":", linewidth=1)
    axes[0].set_title(
        "시장별 일별 등락률 중심 분포 (IQR 및 중앙값)",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].set_ylabel("등락률 (%)")
    axes[0].set_xlabel("시장 구분 (Market Category)")

    # 2) 시장별 거래대금 Log10 박스플롯
    sample_box = df.sample(min(30000, len(df)), random_state=42).copy()
    sample_box["log_trPrc"] = np.log10(sample_box["trPrc"].replace(0, np.nan))

    sns.boxplot(
        data=sample_box,
        x="mrktCtg",
        y="log_trPrc",
        order=market_order,
        showfliers=False,
        palette=["#4C72B0", "#55A868", "#C44E52"],
        ax=axes[1],
        width=0.5,
    )
    axes[1].set_title(
        "시장별 일 거래대금 수준 (Log10 trPrc)", fontsize=13, fontweight="bold"
    )
    axes[1].set_ylabel("Log10(거래대금, 원)")
    axes[1].set_xlabel("시장 구분 (Market Category)")

    plt.tight_layout()
    plt.savefig(
        CHART_DIR / "chart3_market_boxplots.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # ----------------------------------------------------
    # [차트 4] 시계열 시장 거래대금 추이 및 20일 이평선
    # ----------------------------------------------------
    daily_tr = df.groupby("basDt")["trPrc"].sum().reset_index()
    daily_tr["ma20"] = daily_tr["trPrc"].rolling(window=20).mean()

    plt.figure(figsize=(14, 5))
    plt.plot(
        daily_tr["basDt"],
        daily_tr["trPrc"] / 1e12,
        color="lightsteelblue",
        alpha=0.8,
        label="일별 총 거래대금",
    )
    plt.plot(
        daily_tr["basDt"],
        daily_tr["ma20"] / 1e12,
        color="crimson",
        linewidth=2.5,
        label="20일 이동평균선 (MA20)",
    )
    plt.title(
        "2025년 한국 주식시장 일별 거래대금 추이 및 20일 이동평균선",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("날짜 (Date)")
    plt.ylabel("거래대금 (조 원)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(
        CHART_DIR / "chart4_market_timeseries_ma20.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ----------------------------------------------------
    # [차트 5] 당일 시가 vs 종가 (섹터별 양봉 vs 음봉 비율)
    # ----------------------------------------------------
    valid_df = df[(df["mkp"] > 0) & (df["clpr"] > 0) & (df["trqu"] > 0)].copy()
    valid_df["is_yang"] = valid_df["clpr"] > valid_df["mkp"]
    valid_df["is_eum"] = valid_df["clpr"] < valid_df["mkp"]

    candle_stats = (
        valid_df.groupby("sector_macro")
        .agg(
            양봉=("is_yang", lambda x: (x.sum() / len(x)) * 100),
            음봉=("is_eum", lambda x: (x.sum() / len(x)) * 100),
        )
        .sort_values(by="양봉", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    # height -> width=0.6 으로 수정 완료
    candle_stats.plot(
        kind="barh",
        stacked=True,
        color=["#D95F02", "#7570B3"],
        edgecolor="black",
        width=0.6,
        ax=ax,
    )
    ax.axvline(
        50, color="black", linestyle="--", linewidth=1.2, label="50% 기준선"
    )
    ax.set_title(
        "12대 섹터별 당일 캔들 형태 비율 (양봉 vs 음봉: 전 섹터 장중 음봉 우세)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("비율 (%)")
    ax.set_ylabel("12대 대분류 섹터")
    ax.legend(
        ["50% 기준선", "양봉 (종가 > 시가)", "음봉 (종가 < 시가)"],
        loc="lower right",
    )
    plt.tight_layout()
    plt.savefig(
        CHART_DIR / "chart5_sector_and_intraday_momentum.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"🎉 [5/5 완료] 차트 5종이 모두 정상 저장되었습니다:\n➔ {CHART_DIR.resolve()}"
    )


if __name__ == "__main__":
    generate_all_report_charts()