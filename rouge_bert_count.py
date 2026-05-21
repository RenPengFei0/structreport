import pandas as pd
from rouge_score import rouge_scorer
from bert_score import BERTScorer

# 读取数据
df = pd.read_excel('DeepSeek-r1.xlsx')  # 替换为你的Excel文件路径

# 初始化 ROUGE 和 BERTScore 计算器
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
bert_scorer = BERTScorer(lang="zh")

# 计算每个样本的 ROUGE 和 BERTScore
rouge_scores = []
bert_scores = []

for _, row in df.iterrows():
    original_report = str(row['原始报告'])
    structured_report = str(row['结构化报告'])
    
    # 计算 ROUGE 分数
    rouge_score = scorer.score(target=original_report, prediction=structured_report)
    rouge_scores.append({
        'rouge1': rouge_score['rouge1'].fmeasure,
        'rouge2': rouge_score['rouge2'].fmeasure,
        'rougeL': rouge_score['rougeL'].fmeasure
    })
    
    # 计算 BERTScore 分数
    P, R, F1 = bert_scorer.score([structured_report], [original_report])
    bert_scores.append({
        'P': P.item(),
        'R': R.item(),
        'F1': F1.item()
    })

# 将结果添加到数据框中
df['ROUGE1'] = [score['rouge1'] for score in rouge_scores]
df['ROUGE2'] = [score['rouge2'] for score in rouge_scores]
df['ROUGEL'] = [score['rougeL'] for score in rouge_scores]
df['BERTScore_P'] = [score['P'] for score in bert_scores]
df['BERTScore_R'] = [score['R'] for score in bert_scores]
df['BERTScore_F1'] = [score['F1'] for score in bert_scores]

# 保存结果到新的Excel文件
df.to_excel('scores_deepseek.xlsx', index=False)

# 计算平均分数并打印
avg_rouge1 = df['ROUGE1'].mean()
avg_rouge2 = df['ROUGE2'].mean()
avg_rougeL = df['ROUGEL'].mean()
avg_bert_f1 = df['BERTScore_F1'].mean()

print(f"平均 ROUGE-1 分数: {avg_rouge1}")
print(f"平均 ROUGE-2 分数: {avg_rouge2}")
print(f"平均 ROUGE-L 分数: {avg_rougeL}")
print(f"平均 BERTScore F1 分数: {avg_bert_f1}")