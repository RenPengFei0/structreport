import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations


# ==========================
# 1. 读取Excel
# ==========================
file_path = "E:\\研三\\小论文\\all.xlsx"
df = pd.read_excel(file_path)

models = ["Free-text report", "Deepseek", "Gemini", "Gpt-4", "Gpt-4o", "Qwen"]


# ==========================
# 2. 中文字符统计函数
# ==========================
def count_chinese_chars(text):
    if pd.isna(text):
        return 0
    return len(re.findall(r'[\u4e00-\u9fff]', str(text)))


# ==========================
# 3. 统计字数（新版安全写法）
# ==========================
word_df = df[models].apply(lambda col: col.map(count_chinese_chars))


# ==========================
# 4. 描述统计（新版安全写法）
# ==========================
desc_stats = pd.DataFrame({
    "中位数": word_df.median(),
    "平均数": word_df.mean(),
    "标准差": word_df.std(),
    "Q1": word_df.quantile(0.25),
    "Q3": word_df.quantile(0.75),
    "样本量": word_df.count()
})

print("\n===== 描述统计 =====")
print(desc_stats)


# ==========================
# 5. Friedman检验
# ==========================
friedman_stat, friedman_p = friedmanchisquare(
    *[word_df[col] for col in models]
)

print("\n===== Friedman检验 =====")
print(f"统计量 = {friedman_stat:.4f}")
print(f"P值 = {friedman_p:.6f}")


# ==========================
# 6. 两两Wilcoxon比较
# ==========================
pairwise_results = []

pairs = list(combinations(models, 2))
bonf_alpha = 0.05 / len(pairs)

for m1, m2 in pairs:

    stat, p = wilcoxon(word_df[m1], word_df[m2])

    pairwise_results.append([m1, m2, stat, p, p < bonf_alpha])

pairwise_df = pd.DataFrame(pairwise_results,
                           columns=["模型1", "模型2", "统计量", "P值", "Bonferroni显著"])

print("\n===== 两两比较 =====")
print(pairwise_df)


# ==========================
# 7. 转成长格式用于绘图
# ==========================
long_df = word_df.melt(var_name="model", value_name="count")


# ==========================
# 8. 可视化
# ==========================
sns.set(style="whitegrid", font="SimHei")

plt.figure(figsize=(10,6))
sns.boxplot(data=long_df, x="model", y="count")
plt.title("不同模型生成报告字数分布")
plt.show()

plt.figure(figsize=(10,6))
sns.violinplot(data=long_df, x="model", y="count")
plt.title("字数分布密度")
plt.show()


# ==========================
# 9. 导出结果
# ==========================
with pd.ExcelWriter("字数统计结果.xlsx") as writer:
    desc_stats.to_excel(writer, sheet_name="描述统计")
    pairwise_df.to_excel(writer, sheet_name="两两比较")

print("\n统计结果已导出")
