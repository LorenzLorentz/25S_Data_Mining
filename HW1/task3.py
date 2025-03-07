import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

df = pd.read_csv('data_10000.csv')

# TASK 3.1
work_year_cnt=df['work_year'].value_counts().sort_index()
employer_type_cnt=df['employer_type'].value_counts()

plt.subplot(1, 2, 1)
plt.bar(work_year_cnt.index, work_year_cnt.values)
plt.title('Distribution of Work Year')
plt.xlabel('Work Year')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.bar(employer_type_cnt.index, employer_type_cnt.values)
plt.title('Distribution of Employer Type')
plt.xlabel('Employer Type')
plt.ylabel('Count')
plt.xticks(fontproperties=font_manager.FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("task3_year&typedist.png")

# TASK 3.2
percentiles=np.linspace(0, 1, 11)
bins=df['monthly_payment'].quantile(percentiles).values
df['monthly_payment_bins']=pd.cut(df['monthly_payment'], bins, include_lowest=True)
bin_counts=df['monthly_payment_bins'].value_counts().sort_index()
plt.figure(figsize=(10, 10))
plt.bar(bin_counts.index.astype(str), bin_counts.values)
plt.title('Equal Depth Binning of Monthly Payment')
plt.xlabel('Monthly Payment Range')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig("task3_monthlypayment.png")

# TASK 3.3
bins=np.linspace(df['scoring_low'].min(), df['scoring_low'].max(), 11)
df['scoring_low_bins']=pd.cut(df['scoring_low'], bins, include_lowest=True)
bin_counts=df['scoring_low_bins'].value_counts().sort_index()
plt.figure(figsize=(10, 10))
plt.bar(bin_counts.index.astype(str), bin_counts.values, color='teal')
plt.title('Equal Width Binning of Scoring Low')
plt.xlabel('Scoring Low Range')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig("task3_scoringlow.png")