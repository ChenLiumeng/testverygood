from unittest import result
import pandas as pd
import matplotlib.pyplot as plt
import cntext as ct
#print(ct.__version__)
#print(ct.load_pkl_dict('DUTIR.pkl'))

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

file_path = r'C:\Users\clm13\Desktop\try\中国对联_demo.csv'
data = pd.read_csv(file_path)

shanglian_len = data['上联'].astype(str).str.len()
xialian_len = data['下联'].astype(str).str.len()

shanglian_counts = shanglian_len.value_counts().sort_index()
xialian_counts = xialian_len.value_counts().sort_index()

def plot_couplet_length(shanglian, xialian):
    plt.figure(figsize=(10, 6))
    plt.bar(shanglian.index - 0.2, shanglian.values, width=0.4, label='上联', alpha=0.7)
    plt.bar(xialian.index + 0.2, xialian.values, width=0.4, label='下联', alpha=0.7)
    plt.xlabel('字数')
    plt.ylabel('数量')
    plt.title('对联上下联字数分布')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_couplet_length(shanglian_counts, xialian_counts)

diction=ct.load_pkl_dict('DUTIR.pkl')['DUTIR']

data['总字数'] = shanglian_len + xialian_len
grouped=data.groupby('总字数')

def sentiment_analysis(group):
    text = '；'.join([f"{row['上联']}，{row['下联']}" for _, row in group.iterrows()])
    result = ct.sentiment(text=text,diction=diction, lang='chinese')
    return pd.Series(result)

grouped_result = grouped.apply(sentiment_analysis).reset_index()
print(grouped_result)