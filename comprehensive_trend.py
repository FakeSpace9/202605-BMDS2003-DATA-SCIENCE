import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
print("Loading data...")
df = pd.read_csv('Gold Price.csv')

# Ensure Date is formatted as datetime and sort chronologically
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Set the visual style
sns.set_theme(style="whitegrid")

# 2. Create the Subplots (3 rows, 2 columns)
# Adjusted figsize to be wider (15) and slightly shorter (12) for the new layout
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12), sharex=True)
fig.suptitle('Comprehensive Gold Market Trends Over Time (2014-2026)', fontsize=18, fontweight='bold', y=0.96)

# Flatten the 3x2 axes array into a 1D list so we can loop through it easily
axes = axes.flatten()

# Define the features and the specific colors for each chart
features = ['Price', 'Open', 'High', 'Low', 'Volume', 'Chg%']
colors = ['darkgoldenrod', 'teal', 'green', 'crimson', 'purple', 'gray']

# 3. Plot each feature dynamically
for i, (feature, color) in enumerate(zip(features, colors)):
    sns.lineplot(data=df, x='Date', y=feature, ax=axes[i], color=color, linewidth=1.5)
    axes[i].set_ylabel(feature, fontsize=12, fontweight='bold')
    axes[i].set_xlabel("") # Clear x-labels by default to keep it clean

# Set the x-axis label ("Year") for the bottom two charts (index 4 and 5 in the flattened array)
axes[4].set_xlabel("Year", fontsize=14, fontweight='bold')
axes[5].set_xlabel("Year", fontsize=14, fontweight='bold')

# Adjust layout to prevent overlapping titles and axis labels
plt.tight_layout(rect=[0, 0, 1, 0.94])

# 4. Save the high-quality image
plt.savefig('comprehensive_market_trend_2x3.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Success! The 2x3 trend chart has been saved as 'comprehensive_market_trend_2x3.png'.")