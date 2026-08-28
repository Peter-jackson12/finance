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


def generate_all_report_charts():
    print(f"📊 [차트 생성 시작] 데이터 로딩 중: {CLEANED_PATH.name}")
    df = pd.read_csv(CLEANED_PATH, encoding="utf-8-sig", low_memory=False)
    df["basDt"] = pd.to_datetime(df["basDt"].astype(str))

    # ----------------------------------------------------
    # [차트 1] 시장별 거래 건수 & 12대 섹터별 거래대금 점유율
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    market_counts = df["mrktCtg"].value_counts()
    axes[0].bar(
        market_counts.index,
        market_counts.values,
        color=["#4C72B0", "#55A868", "#C44E52"],
        edgecolor="black",
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
            v + 5000,
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
    sector_tr.plot(kind="barh", ax=axes[1], color="teal", edgecolor="black")
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
    # [차트 2] 일별 등락률 분포 & 시가총액-거래대금 산점도
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
    # [차트 3] 시장별 등락률 및 거래대금 박스플롯
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.boxplot(
        data=df,
        x="mrktCtg",
        y=df["fltRt"].clip(-30, 30),
        ax=axes[0],
        palette="Set2",
    )
    axes[0].set_title(
        "시장별 일별 등락률 분포 및 이상치", fontsize=13, fontweight="bold"
    )
    axes[0].set_ylabel("등락률 (%)")

    sample_box = df.sample(min(20000, len(df)), random_state=42).copy()
    sample_box["log_trPrc"] = np.log10(sample_box["trPrc"].replace(0, np.nan))
    sns.boxplot(
        data=sample_box, x="mrktCtg", y="log_trPrc", ax=axes[1], palette="Set2"
    )
    axes[1].set_title(
        "시장별 거래대금 수준 (Log10 trPrc)", fontsize=13, fontweight="bold"
    )
    axes[1].set_ylabel("Log10(거래대금)")
    plt.tight_layout()
    plt.savefig(
        CHART_DIR / "chart3_market_boxplots.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # ----------------------------------------------------
    # [차트 4] 시계열 일별 거래대금 추이 및 MA20 이동평균선
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
    # [차트 5] 당일 시가 vs 종가 (12대 섹터별 양봉 vs 음봉 비율)
    # ----------------------------------------------------
    valid_df = df[(df["mkp"] > 0) & (df["clpr"] > 0) & (df["trqu"] > 0)].copy()
    valid_df["is_yang"] = valid_df["clpr"] > valid_df["mkp"]
    valid_df["is_eum"] = valid_df["clpr"] < valid_df["mkp"]

    candle_stats = (
        valid_df.groupby("sector_macro")
        .agg(
            양봉=(
                "is_yang",
                lambda x: (x.sum() / len(x)) * 100,
            ),
            음봉=(
                "is_eum",
                lambda x: (x.sum() / len(x)) * 100,
            ),
        )
        .sort_values(by="양봉", ascending=True)
    )

    plt.figure(figsize=(12, 6))
    candle_stats.plot(
        kind="barh",
        stacked=True,
        color=["#D95F02", "#7570B3"],
        figsize=(12, 6),
        edgecolor="black",
    )
    plt.axvline(
        50, color="black", linestyle="--", linewidth=1, label="50% 기준선"
    )
    plt.title(
        "12대 섹터별 당일 캔들 형태 비율 (양봉 vs 음봉: 장중 음봉 우세 확인)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("비율 (%)")
    plt.ylabel("12대 대분류 섹터")
    plt.legend(["50% 기준", "양봉 (종가 > 시가)", "음봉 (종가 < 시가)"], loc="lower right")
    plt.tight_layout()
    plt.savefig(
        CHART_DIR / "chart5_sector_and_intraday_momentum.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"🎉 [완료] 차트 5종이 정상 저장되었습니다: {CHART_DIR.resolve()}")


if __name__ == "__main__":
    generate_all_report_charts()