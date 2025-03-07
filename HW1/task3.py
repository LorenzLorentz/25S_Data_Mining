import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

df = pd.read_csv('data_10000.csv')

# TASK 3.1
work_year_cnt=df['work_year'].value_counts().sort_index()
employer_type_cnt=df['employer_type'].value_counts().sort_index()

plt.subplot(1, 2, 1)
plt.bar(work_year_cnt.index.astype(str), work_year_cnt.values)
plt.title('Distribution of Work Year')
plt.xlabel('Work Year')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.bar(employer_type_cnt.index.astype(str), employer_type_cnt.values)
plt.title('Distribution of Employer Type')
plt.xlabel('Employer Type')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("task3_year&typedist.png")

# TASK 3.2
# TASK 3.3