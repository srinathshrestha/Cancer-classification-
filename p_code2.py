import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


csv_file = "Cancer_data.csv"
df = pd.read_csv(csv_file)
df.drop("Unnamed: 32", axis=1, inplace=True)


malignant_df = df[df["Diagnosis Found"] == "M"]
benign_df = df[df["Diagnosis Found"] == "B"]


columns_to_exclude = ["Cancer_id", "Diagnosis Found"]
malignant_df = malignant_df.drop(columns=columns_to_exclude)
benign_df = benign_df.drop(columns=columns_to_exclude)

#Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(malignant_df.astype(float), cmap="coolwarm", annot=True, fmt=".1f")
plt.title('Heatmap of Attributes for "M" Diagnoses')
plt.show()
