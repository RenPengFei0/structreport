import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


file_paths = [
    "",  # 模型1（如Deepseek）的Excel路径
    "",  # 模型2（如GPT-4）的Excel路径
    "",  # 模型3的Excel路径
    "",  # 模型4的Excel路径
    ""  # 模型5的Excel路径
]
model_names = ["model1", "model2", "model3", "model4", "model5"]  # 模型名称
target_metrics = ["ROUGE1", "ROUGE2", "ROUGEL", "BERTScore_P", "BERTScore_R", "BERTScore_F1"]  # 要分析的指标
# --------------------------

# 2. 读取所有模型的Excel数据
model_data = {}
for path, name in zip(file_paths, model_names):
    df = pd.read_excel(path)
    # 提取目标指标的列（确保Excel中列名与target_metrics一致）
    model_data[name] = df[target_metrics]
    print(f"已读取{name}：{len(model_data[name])}条数据")  # 验证是否是644条

# 3. 对每个指标单独计算p值
for metric in target_metrics:
    print(f"\n===== 分析指标：{metric} =====")

    # 收集5个模型的该指标数据
    metric_datas = [model_data[name][metric].values for name in model_names]

    # 3.1 单因素ANOVA（大样本无需验证正态性）
    anova_stat, anova_p = stats.f_oneway(*metric_datas)
    print(f"整体差异p值：{anova_p:.6f} → {'存在显著差异' if anova_p <= 0.05 else '无显著差异'}")

    # 3.2 若整体显著，做Tukey HSD两两比较
    if anova_p <= 0.05:
        # 整理成检验需要的格式
        all_values = np.concatenate(metric_datas)
        all_groups = np.repeat(model_names, [len(data) for data in metric_datas])

        # 执行Tukey HSD
        tukey_result = pairwise_tukeyhsd(endog=all_values, groups=all_groups, alpha=0.05)
        print("\n两两模型差异显著的配对（p≤0.05）：")
        # 提取显著的配对
        for pair in tukey_result.summary().data[1:]:
            if float(pair[4]) <= 0.05:
                print(f"{pair[0]} vs {pair[1]}: p={float(pair[4]):.4f}")