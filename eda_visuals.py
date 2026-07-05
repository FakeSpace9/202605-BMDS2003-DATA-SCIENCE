import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Generating Data Visualisations...")

# 1. Load the Data
df = pd.read_csv('Gold Price.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Set aesthetic style
sns.set_theme(style="whitegrid")

# ==========================================
# VISUALISATION 1: Time Series of Gold Price
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Price'], color='gold', linewidth=1.5)
plt.title('Daily Closing Price of Gold (2014 - 2026)', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Closing Price', fontsize=12)
plt.fill_between(df['Date'], df['Price'], color='gold', alpha=0.2)
plt.tight_layout()
plt.savefig('gold_price_trend.png')
print("Saved: gold_price_trend.png")

# ==========================================
# VISUALISATION 2: Correlation Heatmap
# ==========================================
plt.figure(figsize=(8, 6))
# Select only numerical columns for correlation
numerical_cols = df[['Price', 'Open', 'High', 'Low', 'Volume', 'Chg%']]
correlation_matrix = numerical_cols.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5)
plt.title('Correlation Heatmap of Trading Metrics', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
print("Saved: correlation_heatmap.png")
print("Visualisations complete! Please insert these images into your Google Docs report.")