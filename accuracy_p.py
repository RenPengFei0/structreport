# ======================================
# 多模型影像特征准确率统计分析（完整版 + 正态检验 + 可视化）
# ======================================

import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, shapiro, norm
import itertools
import matplotlib.pyplot as plt

# ----------------------------
# Step 1 读取数据
# ----------------------------
file_path = ""
df = pd.read_excel(file_path)

models = ["Deepseek","GPT-4","GPT-4o","Qwen","Gemini","Keyword search algorithm"]

# ----------------------------
# Step 2 描述统计
# ----------------------------
summary_df = pd.DataFrame({
    "Model": models,
    "Mean Accuracy":[df[m].mean() for m in models],
    "Std":[df[m].std() for m in models],
    "Median":[df[m].median() for m in models]
})

print("\n===== 描述统计 =====")
print(summary_df)

# ----------------------------
# Step 2.5 正态分布检验
# ----------------------------
normality_results = []
for m in models:
    stat, p = shapiro(df[m])
    normality_results.append([m, stat, p, p > 0.05])

normality_df = pd.DataFrame(
    normality_results,
    columns=["Model", "Shapiro_Stat", "Shapiro_p", "Is_Normal"]
)

print("\n===== Shapiro 正态性检验 =====")
print(normality_df)

# ----------------------------
# Step 2.6 正态分布可视化（新增！）
# ----------------------------
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 直方图 + 正态拟合曲线
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, m in enumerate(models):
    data = df[m].dropna()
    mu, std = norm.fit(data)
    axes[i].hist(data, bins=10, density=True, alpha=0.6, color='skyblue', edgecolor='k')

    xmin, xmax = axes[i].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    y = norm.pdf(x, mu, std)
    axes[i].plot(x, y, 'r--', linewidth=2)
    axes[i].set_title(f"{m}\nShapiro p={normality_df.loc[i,'Shapiro_p']:.3f}")
    axes[i].set_xlabel("Accuracy")

plt.suptitle("各模型准确率分布直方图 + 正态拟合", fontsize=16)
plt.tight_layout()
plt.show()

# 2. Q-Q 图（最能证明是否正态）
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, m in enumerate(models):
    data = df[m].dropna()
    from scipy.stats import probplot
    probplot(data, plot=axes[i])
    axes[i].set_title(f"{m}\nQ-Q Plot")

plt.suptitle("各模型准确率 Q-Q 图（点越贴近直线越正态）", fontsize=16)
plt.tight_layout()
plt.show()

# ----------------------------
# Step 3 Friedman总体检验
# ----------------------------
data = [df[m] for m in models]
friedman_stat, friedman_p = friedmanchisquare(*data)

print("\n===== Friedman检验 =====")
print("Chi-square =", friedman_stat)
print("p-value =", friedman_p)

# ----------------------------
# Step 4 Kendall效应量
# ----------------------------
n = len(df)
k = len(models)
kendall_w = friedman_stat / (n * (k - 1))

print("\n===== Kendall W效应量 =====")
print("Kendall W =", kendall_w)

# ----------------------------
# Step 5 两两Wilcoxon比较
# ----------------------------
pairs = list(itertools.combinations(models, 2))
pairwise_results = []

for m1, m2 in pairs:
    stat, p = wilcoxon(df[m1], df[m2])
    pairwise_results.append([m1, m2, p])

pairwise_df = pd.DataFrame(pairwise_results, columns=["Model1","Model2","Raw_p"])
pairwise_df["Bonferroni_p"] = pairwise_df["Raw_p"] * len(pairwise_df)
pairwise_df["Bonferroni_p"] = pairwise_df["Bonferroni_p"].clip(upper=1)

print("\n===== 两两比较结果 =====")
print(pairwise_df)

# ----------------------------
# Step 6 模型平均排名
# ----------------------------
rank_df = df[models].rank(axis=1, ascending=False)
mean_rank = rank_df.mean().sort_values()

rank_result = pd.DataFrame({
    "Model": mean_rank.index,
    "Average Rank": mean_rank.values
})

print("\n===== 模型平均排名 =====")
print(rank_result)

# ----------------------------
# Step 7 特征难度分析
# ----------------------------
df["Feature_Mean"] = df[models].mean(axis=1)
feature_difficulty = df[["Features","Feature_Mean"]].sort_values("Feature_Mean")

print("\n===== 特征难度排序 =====")
print(feature_difficulty)

# ----------------------------
# Step 8 箱线图
# ----------------------------
plt.figure(figsize=(10,5))
plt.boxplot([df[m] for m in models], labels=models)
plt.ylabel("Accuracy")
plt.title("各模型准确率分布")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# ----------------------------
# Step 9 平均准确率柱状图
# ----------------------------
means = [df[m].mean() for m in models]
plt.figure(figsize=(10,5))
plt.bar(models, means)
plt.ylabel("Mean Accuracy")
plt.title("模型平均准确率对比")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# ----------------------------
# Step 10 模型排名图
# ----------------------------
plt.figure(figsize=(10,5))
plt.bar(rank_result["Model"], rank_result["Average Rank"])
plt.ylabel("Average Rank")
plt.title("模型平均排名")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# ----------------------------
# Step 11 导出Excel
# ----------------------------
with pd.ExcelWriter("") as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    normality_df.to_excel(writer, sheet_name="Normality_Test", index=False)
    pd.DataFrame({
        "Friedman_Chi_square":[friedman_stat],
        "p_value":[friedman_p],
        "Kendall_W":[kendall_w]
    }).to_excel(writer, sheet_name="Friedman_Result", index=False)
    pairwise_df.to_excel(writer, sheet_name="Pairwise_Test", index=False)
    rank_result.to_excel(writer, sheet_name="Model_Rank", index=False)
    feature_difficulty.to_excel(writer, sheet_name="Feature_Difficulty", index=False)

print("\n统计分析结果已保存到：统计分析结果.xlsx")