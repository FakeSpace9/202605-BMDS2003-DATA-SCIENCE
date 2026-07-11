import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================
print("Loading data...")
df = pd.read_csv('Gold Price_MYR.csv')

# Format Date for time-series charts
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Create Market Trend category for the 2-Color Pairplot
df['Market Trend'] = df['Chg%'].apply(lambda x: 'Up' if x > 0 else 'Down')

# Set professional visual style 
sns.set_theme(style="whitegrid")

# ==========================================
# VISUAL 1: Comprehensive Trend (2x3 Grid)
# ==========================================
print("Generating 2x3 Comprehensive Trend Chart...")
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12), sharex=True)
fig.suptitle('Comprehensive Gold Market Trends Over Time (MYR) (2014-2026)', fontsize=18, fontweight='bold', y=0.96)

axes = axes.flatten()
features = ['Price', 'Open', 'High', 'Low', 'Volume', 'Chg%']
colors = ['darkgoldenrod', 'teal', 'green', 'crimson', 'purple', 'gray']

for i, (feature, color) in enumerate(zip(features, colors)):
    sns.lineplot(data=df, x='Date', y=feature, ax=axes[i], color=color, linewidth=1.5)
    axes[i].set_ylabel(feature, fontsize=12, fontweight='bold')
    axes[i].set_xlabel("") 

axes[4].set_xlabel("Year", fontsize=14, fontweight='bold')
axes[5].set_xlabel("Year", fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('comprehensive_market_trend_2x3.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# VISUAL 2: Correlation Heatmap
# ==========================================
print("Generating Correlation Heatmap...")
plt.figure(figsize=(8, 6))
numerical_df = df[['Price', 'Open', 'High', 'Low', 'Volume', 'Chg%']]
correlation_matrix = numerical_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5)
plt.title('Correlation Heatmap of Trading Metrics', fontsize=14, fontweight='bold')
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# VISUAL 3: Horizontal Boxplot 
# ==========================================
print("Generating Horizontal Boxplot...")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['Open', 'High', 'Low']], orient='h', palette="Set2")
plt.title('Boxplot Analysis of Intra-day Trading Features (MYR)', fontsize=14, fontweight='bold')
plt.xlabel('Price (MYR)', fontsize=12) 
plt.ylabel('Market Features', fontsize=12)
plt.savefig('boxplot_features.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# VISUAL 4: 2-Color Pairplot 
# ==========================================
print("Generating 2-Color Pairplot... (This might take a minute)")
pair_plot = sns.pairplot(
    df[['Open', 'High', 'Low', 'Volume', 'Price', 'Market Trend']], 
    hue='Market Trend',
    palette={'Up': 'seagreen', 'Down': 'indianred'},
    diag_kind='kde',
    plot_kws={'alpha': 0.5, 'edgecolor': None}
)
pair_plot.fig.suptitle('Pairplot of Gold Market Variables by Daily Trend', y=1.02, fontsize=16, fontweight='bold')
pair_plot.savefig('pairplot_analysis_2color.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# VISUAL 5: 30-Day Rolling Volatility
# ==========================================
print("Generating 30-Day Rolling Volatility...")
# Calculate daily returns and rolling standard deviation
df['Return'] = df['Price'].pct_change() * 100
df['Volatility_30d'] = df['Return'].rolling(window=30).std()

fig2, ax2 = plt.subplots(figsize=(11, 5))
ax2.plot(df['Date'], df['Volatility_30d'], color='#8B0000', linewidth=1.2)
ax2.fill_between(df['Date'], df['Volatility_30d'], color='#8B0000', alpha=0.15)
ax2.set_title('Gold Price: 30-Day Rolling Volatility (Std Dev of Daily Returns)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Volatility (%)')
ax2.grid(alpha=0.3)
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.savefig('chart2_rolling_volatility.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Success! All 5 master visuals (including Volatility) have been saved to your project folder.")