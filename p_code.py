import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


csv_file = "Cancer_data.csv"
df = pd.read_csv(csv_file)
df.drop("Unnamed: 32", axis=1, inplace=True)


df["Diagnosis Found"] = df["Diagnosis Found"].map({"M": 1, "B": 0})


columns_to_exclude = ["Cancer_id", "Diagnosis Found"]
correlation_matrix = df.drop(columns=columns_to_exclude).corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".1f")
plt.title("Correlation Heatmap (All Columns except 'Cancer_id' and 'Diagnosis Found')")
plt.show()


diagnosis_M = df[df["Diagnosis Found"] == 1]
diagnosis_B = df[df["Diagnosis Found"] == 0]

for column_name in df.columns:
    if column_name not in columns_to_exclude:
        plt.figure(figsize=(8, 6))
        sns.histplot(diagnosis_M[column_name], color="red", label="M", alpha=0.5)
        sns.histplot(diagnosis_B[column_name], color="blue", label="B", alpha=0.5)
        plt.xlabel(column_name)
        plt.ylabel("Frequency")
        plt.title(f"{column_name} distribution for 'M' vs. 'B' diagnoses")
        plt.legend()
        plt.show()
