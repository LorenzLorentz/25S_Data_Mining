import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

df = pd.read_csv('data_10000.csv')

# TASK 2.1
mean_value=df['total_loan'].mean()
df['total_loan']=df['total_loan'].fillna(mean_value)
df.to_csv('data_10000.csv')

# TASK 2.2
five_number=df['total_loan'].describe(percentiles=[0.25, 0.5, 0.75])
five_number_table = pd.DataFrame({
    "min": [five_number['min']],
    "25%": [five_number['25%']],
    "50%": [five_number['50%']],
    "50%": [five_number['50%']],
    "max": [five_number['max']],
})
plt.table(cellText=five_number_table.values, colLabels=five_number_table.columns, loc='center', cellLoc='center')
plt.axis('off')
plt.savefig("task2_fivenumber.png")
plt.close()

plt.boxplot(df['total_loan'])
plt.title('Boxplot of Total Loan')
plt.ylabel('Total Loan')
plt.savefig("task2_boxplot.png")
plt.close()

# TASK 2.3
plt.scatter(df['scoring_low'], df['interest'], label='Data points')
plt.ylim(0, 35)
plt.xlim(600, 900)
coefficients=np.polyfit(df['scoring_low'], df['interest'], 1)
linear_fit=np.poly1d(coefficients)
x_values=np.linspace(640, df['scoring_low'].max(), 100)
plt.plot(x_values, linear_fit(x_values), color='red', label='Regression line')
plt.title('Scatter Plot of Interest vs Scoring Low')
plt.xlabel('Scoring Low')
plt.ylabel('Interest')
plt.savefig("task2_sca_int&scorelow.png")
plt.close()

pearson_corr, p_value = stats.pearsonr(df['scoring_low'].dropna(), df['interest'].dropna())
corr_table = pd.DataFrame({
    "pearson_corr": [pearson_corr],
    "p_value": [p_value],
})
plt.table(cellText=corr_table.values, colLabels=corr_table.columns, loc='center', cellLoc='center')
plt.axis('off')
plt.savefig("task2_corrtable.png")
plt.close()