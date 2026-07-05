import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the dataset
print("Loading data...")
df = pd.read_csv('Gold Price.csv')

# Ensure Date is formatted correctly for the trend line
df['Date'] = pd.to_datetime(df['Date'])

# Set the visual style 
sns.set_theme(style="whitegrid")

# ==========================================
# VISUAL 1: Gold Price Trend (Line Chart)
# ==========================================
print("Generating Gold Price Trend...")
plt.figure(figsize=(12, 6))
# Sorting data chronologically is important for line charts
df_sorted = df.sort_values('Date')
sns.lineplot(data=df_sorted, x='Date', y='Price', color='darkgoldenrod')
plt.title('Daily Closing Price of Gold (2014 - 2026)', fontsize=14, fontweight='bold')
plt.ylabel('Price', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.savefig('gold_price_trend.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# VISUAL 2: Correlation Heatmap
# ==========================================
print("Generating Correlation Heatmap...")
plt.figure(figsize=(8, 6))
# Select only the numerical columns to correlate
numerical_df = df[['Price', 'Open', 'High', 'Low', 'Volume', 'Chg%']]
correlation_matrix = numerical_df.corr()
# Draw the heatmap with the correlation numbers annotated
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5)
plt.title('Correlation Heatmap of Trading Metrics', fontsize=14, fontweight='bold')
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# VISUAL 3: Boxplot for Feature Outliers
# ==========================================
print("Generating Boxplot...")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['Open', 'High', 'Low']], palette="Set2")
plt.title('Boxplot Analysis of Intra-day Trading Features', fontsize=14, fontweight='bold')
plt.ylabel('Price', fontsize=12)
plt.xlabel('Market Features', fontsize=12)
plt.savefig('boxplot_features.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# VISUAL 4: Pairplot for Feature Relationships
# ==========================================
print("Generating Pairplot... (This might take a few seconds)")
pair_plot = sns.pairplot(
    df[['Open', 'High', 'Low', 'Volume', 'Price']], 
    diag_kind='kde',
    plot_kws={'alpha': 0.6, 'edgecolor': None}
)
pair_plot.fig.suptitle('Pairplot of Gold Market Variables', y=1.02, fontsize=16, fontweight='bold')
pair_plot.savefig('pairplot_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Success! All 4 visuals have been saved to your project folder.")