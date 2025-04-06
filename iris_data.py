# iris_data_analysis.py

# Objective:
# - Load and analyze the Iris dataset using pandas
# - Create simple plots with matplotlib for visualization

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# === Load Dataset ===
try:
    iris = load_iris(as_frame=True)
    df = iris.frame
    print("✅ Dataset loaded successfully!\n")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# === Explore the Dataset ===
print("🔍 First 5 rows of the dataset:")
print(df.head())

print("\n📊 Dataset Info:")
print(df.info())

print("\n🧼 Missing values check:")
print(df.isnull().sum())

# === Clean Dataset ===
df_cleaned = df.dropna()

# Add species column
df_cleaned['species'] = df_cleaned['target'].map(dict(zip(range(3), iris.target_names)))

# === Basic Data Analysis ===
print("\n📈 Descriptive Statistics:")
print(df_cleaned.describe())

print("\n📊 Mean of each feature grouped by species:")
print(df_cleaned.groupby('species').mean())

# === Visualizations ===
sns.set(style="whitegrid")

# Line Chart - Cumulative petal length
plt.figure(figsize=(10, 5))
df_cleaned['petal length (cm)'].cumsum().plot(kind='line')
plt.title('Cumulative Petal Length Over Entries')
plt.xlabel('Entry Index')
plt.ylabel('Cumulative Petal Length (cm)')
plt.grid(True)
plt.savefig("line_chart.png")
plt.show()

# Bar Chart - Average petal length per species
plt.figure(figsize=(8, 5))
sns.barplot(x='species', y='petal length (cm)', data=df_cleaned, ci=None)
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.savefig("bar_chart.png")
plt.show()

# Histogram - Sepal width distribution
plt.figure(figsize=(8, 5))
sns.histplot(df_cleaned['sepal width (cm)'], bins=15, kde=True)
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.savefig("histogram.png")
plt.show()

# Scatter Plot - Sepal length vs Petal length
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df_cleaned)
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.savefig("scatter_plot.png")
plt.show()

# === Observations ===
print("\n📝 Observations:")
print("- Setosa has significantly smaller petal length and width.")
print("- Scatter plot shows clear separation of Setosa from other species.")
print("- Distribution of sepal width shows a slightly right-skewed pattern.")

# End of script
