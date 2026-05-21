# ===================== 正态分布检验完整代码 =====================
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ---------------------- 1. 读取 Excel 数据 ----------------------
# 【注意】把这里改成你的 Excel 文件路径！
# Windows 示例：r"C:\Users\xxx\Desktop\data.xlsx"
# Mac 示例："/Users/xxx/Desktop/data.xlsx"
# file_path = "report_with_scores_gemini.xlsx"
file_path = "report_with_scores_guanjianci.xlsx"
# file_path = "report_with_scores_qwen.xlsx"
# file_path = "report_with_scores_gpt4o.xlsx"
# file_path = "report_with_scores_gpt4.xlsx"

# 读取数据（自动识别表头）
df = pd.read_excel(file_path)

# 你需要检验的指标（与你提供的表头完全一致）
target_columns = [
    "ROUGE1",
    "ROUGE2",
    "ROUGEL",
    "BERTScore_P",
    "BERTScore_R",
    "BERTScore_F1"
]


# ---------------------- 2. 正态分布检验函数 ----------------------
def normality_test(data, col_name):
    """
    对一列数据进行正态分布检验
    返回：是否符合正态分布（p>0.05 视为符合）
    """
    # 去除空值
    data = data.dropna()

    print(f"\n{'=' * 50}")
    print(f"📊 指标：{col_name}")
    print(f"样本数量：{len(data)}")

    # ========== 3种常用正态分布检验 ==========
    # 1. Shapiro-Wilk 检验（小样本推荐，n<5000）
    shapiro_stat, shapiro_p = stats.shapiro(data)
    print(f"Shapiro检验 p值：{shapiro_p:.4f}")

    # 2. Kolmogorov-Smirnov 检验（大样本）
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
    print(f"KS检验 p值：{ks_p:.4f}")

    # 3. 正态性判断标准：p > 0.05 视为符合正态分布
    is_normal = shapiro_p > 0.05
    print(f"✅ 是否符合正态分布：{'是' if is_normal else '否'}")

    return is_normal


# ---------------------- 3. 批量检验所有指标 ----------------------
print("🔍 开始正态分布检验")
results = {}
for col in target_columns:
    if col in df.columns:
        results[col] = normality_test(df[col], col)
    else:
        print(f"\n⚠️ 未找到指标：{col}")

# ---------------------- 4. 绘制分布图（直观判断） ----------------------
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]  # 正常显示中文
plt.rcParams["axes.unicode_minus"] = False

# 为每个指标绘制直方图+正态拟合曲线
for col in target_columns:
    if col not in df.columns:
        continue

    data = df[col].dropna()
    plt.figure(figsize=(8, 4))

    # 绘制直方图
    n, bins, patches = plt.hist(data, bins=15, density=True, alpha=0.6, color='skyblue')

    # 绘制正态分布拟合曲线
    mu, sigma = np.mean(data), np.std(data)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='正态分布拟合曲线')

    plt.title(f"{col} 分布直方图", fontsize=12)
    plt.xlabel("数值")
    plt.ylabel("密度")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# ---------------------- 5. 最终汇总结果 ----------------------
print(f"\n{'=' * 50}")
print("📋 所有指标正态分布检验汇总")
for col, res in results.items():
    print(f"{col:<15}：{'✅ 符合' if res else '❌ 不符合'}")