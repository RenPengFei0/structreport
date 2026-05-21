# ===============================
# 1. 导入库
# ===============================

import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ===============================
# 2. 文件路径（修改为你自己的路径）
# ===============================

files = {
    "Deepseek": "",
    "GPT-4": "",
    "GPT-4o": "",
    "Qwen": "",
    "Gemini": "",
    "Keyword": ""
}


# ===============================
# 3. 指标列名映射
# ===============================

metrics = {
    "ROUGE-1": "ROUGE1",
    "ROUGE-2": "ROUGE2",
    "ROUGE-L": "ROUGEL",
    "BERTScore-P": "BERTScore_P",
    "BERTScore-R": "BERTScore_R",
    "BERTScore-F1": "BERTScore_F1"
}


# ===============================
# 4. 读取数据
# ===============================

data = {}
for model, path in files.items():
    data[model] = pd.read_excel(path)


# ===============================
# 5. 对齐数据
# ===============================

aligned = {}

for metric, col in metrics.items():

    df = pd.DataFrame({
        model: data[model][col]
        for model in files
    })

    aligned[metric] = df.dropna()


# ===============================
# 6. Friedman检验 + Kendall W
# ===============================

friedman_results = []

for metric, df in aligned.items():

    stat, p = friedmanchisquare(*[df[c] for c in df.columns])

    n, k = df.shape
    W = stat / (n * (k - 1))

    friedman_results.append([metric, stat, p, W])

friedman_table = pd.DataFrame(
    friedman_results,
    columns=["Metric", "Chi-square", "P-value", "Kendall_W"]
)

print("\n===== Friedman检验结果 =====")
print(friedman_table)


# ===============================
# 7. Wilcoxon两两比较 + Bonferroni
# ===============================

pairwise_results = []

for metric, df in aligned.items():

    keyword = df["Keyword"]
    models = [m for m in df.columns if m != "Keyword"]

    for model in models:

        stat, p = wilcoxon(df[model], keyword)
        p_corrected = min(p * len(models), 1)

        pairwise_results.append([metric, model, p, p_corrected])

pairwise_table = pd.DataFrame(
    pairwise_results,
    columns=["Metric", "Model_vs_Keyword", "Raw_P", "Bonferroni_P"]
)

print("\n===== Wilcoxon两两比较结果 =====")
print(pairwise_table)


# ===============================
# 8. Nemenyi事后检验
# ===============================

nemenyi_results = {}

for metric, df in aligned.items():

    nemenyi = sp.posthoc_nemenyi_friedman(df)
    nemenyi_results[metric] = nemenyi

    print(f"\n===== {metric} Nemenyi检验 =====")
    print(nemenyi)


# ===============================
# 9. 创建图像保存文件夹
# ===============================

os.makedirs("stat_plots", exist_ok=True)


# ===============================
# 10. 箱线图
# ===============================

for metric, df in aligned.items():

    plt.figure(figsize=(8,6))
    sns.boxplot(data=df)

    plt.title(metric)
    plt.ylabel("Score")

    plt.tight_layout()
    plt.savefig(f"stat_plots/{metric}_boxplot.png")
    plt.close()


# ===============================
# 11. 平均性能柱状图
# ===============================

for metric, df in aligned.items():

    mean_scores = df.mean().sort_values(ascending=False)

    plt.figure(figsize=(8,6))
    mean_scores.plot(kind="bar")

    plt.title(f"Average {metric}")
    plt.ylabel("Score")

    plt.tight_layout()
    plt.savefig(f"stat_plots/{metric}_mean_bar.png")
    plt.close()


# ===============================
# 12. 模型平均排名图
# ===============================

for metric, df in aligned.items():

    ranks = df.rank(axis=1, ascending=False)
    avg_rank = ranks.mean().sort_values()

    plt.figure(figsize=(8,6))
    avg_rank.plot(kind="bar")

    plt.title(f"Average Rank ({metric})")
    plt.ylabel("Rank (Lower is better)")

    plt.tight_layout()
    plt.savefig(f"stat_plots/{metric}_rank_bar.png")
    plt.close()


# ===============================
# 13. 导出统计结果Excel
# ===============================

with pd.ExcelWriter("statistical_results.xlsx") as writer:

    friedman_table.to_excel(writer, sheet_name="Friedman", index=False)
    pairwise_table.to_excel(writer, sheet_name="Wilcoxon", index=False)

    for metric, table in nemenyi_results.items():
        table.to_excel(writer, sheet_name=f"Nemenyi_{metric}")


print("\n统计分析完成！结果已保存。")
