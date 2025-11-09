import pandas as pd
import matplotlib.pyplot as plt

dt_politifact = pd.read_csv('politifact_kaggle_dataset_cleaned.csv')
dt_politifact.info()
print(dt_politifact.head())

politicians = dt_politifact['politician_name'].value_counts()

politician_pie, politician_ax = plt.subplots(figsize=(16, 13))
politician_ax.pie(politicians, labels=politicians.index, autopct='%1.1f%%')
politician_ax.set_title('Distribution of Statements by Politician')
plt.show()


sample = dt_politifact.groupby('politician_name', group_keys=False).apply(
    lambda x: x.sample(frac=0.0439))  # 0.155 for ~1000 samples

print(sample.info())

sample.to_csv('politifact_kaggle_dataset_sampled.csv', index=False)

sample_politicians = sample['politician_name'].value_counts()
sample_politician_pie, sample_politician_ax = plt.subplots(figsize=(16, 13))
sample_politician_ax.pie(
    sample_politicians, labels=sample_politicians.index, autopct='%1.1f%%')
sample_politician_ax.set_title(
    'Distribution of Statements by Politician (Sampled)')
plt.show()
