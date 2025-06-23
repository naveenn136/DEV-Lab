import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
titanic = sns.load_dataset('titanic')

print(titanic.head())
print("\nDataset Info:")
print(titanic.info())
print("\nMissing values:")
print(titanic.isnull().sum())
print("\nSummary statistics:")
print(titanic.describe(include='all'))

sns.set(style="darkgrid")
plt.figure(figsize=(6,4))
sns.countplot(data=titanic, x='survived', palette='Set2')
plt.title('Survival Count (0 = No, 1 = Yes)')
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=titanic, x='sex', hue='survived', palette='Set1')
plt.title('Survival by Gender')
plt.show()
# 3. Countplot - Class vs Survival
plt.figure(figsize=(6,4))
sns.countplot(data=titanic, x='class', hue='survived', palette='Set3')
plt.title('Survival by Passenger Class')
plt.show()

# 4. Age Distribution - Histogram
plt.figure(figsize=(8,4))
sns.histplot(data=titanic, x='age', kde=True, bins=30, color='skyblue')
plt.title('Age Distribution')
plt.show()

# 5. Boxplot - Age vs Class
plt.figure(figsize=(6,4))
sns.boxplot(data=titanic, x='class', y='age', palette='Pastel1')
plt.title('Age Distribution by Passenger Class')
plt.show()

# 6. Heatmap - Correlation Matrix
plt.figure(figsize=(8,6))
sns.heatmap(titanic.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 7. Violinplot - Age vs Survival by Gender
plt.figure(figsize=(6,4))
sns.violinplot(data=titanic, x='sex', y='age', hue='survived', split=True, palette='muted')
plt.title('Age Distribution by Gender and Survival')
plt.show()

# 8. Barplot - Embark Town vs Survival Rate
plt.figure(figsize=(6,4))
sns.barplot(data=titanic, x='embark_town', y='survived', palette='spring')
plt.title('Survival Rate by Embark Town')
plt.show()