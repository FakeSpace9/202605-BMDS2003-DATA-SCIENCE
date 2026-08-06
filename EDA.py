import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set global plotting style & layout
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.autolayout': True})

# 1. Load Dataset
df = pd.read_csv('SeoulBikeData.csv', encoding='latin1')

# 2. Preprocessing & Feature Extraction
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.day_name()
df['Is_Weekend'] = df['DayOfWeek'].isin(['Saturday', 'Sunday'])

# Create output directory for saved graphs
os.makedirs('graphs', exist_ok=True)

# --- GRAPH 1: Distribution of Rented Bike Count ---
plt.figure(figsize=(10, 6))
sns.histplot(df['Rented Bike Count'], kde=True, color='teal', bins=30)
plt.title('Graph 1: Distribution of Rented Bike Count', fontsize=14, fontweight='bold')
plt.xlabel('Rented Bike Count', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.savefig('graphs/graph_1.png', dpi=300)
plt.close()

# --- GRAPH 2: Hourly Trend of Bike Rentals Across Seasons ---
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Hour', y='Rented Bike Count', hue='Seasons', marker='o', palette='Set2')
plt.title('Graph 2: Hourly Trend of Bike Rentals Across Seasons', fontsize=14, fontweight='bold')
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Average Rented Bike Count', fontsize=12)
plt.legend(title='Seasons')
plt.savefig('graphs/graph_2.png', dpi=300)
plt.close()

# --- GRAPH 3: Rented Bike Count Distribution by Season ---
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Seasons', y='Rented Bike Count', palette='pastel', order=['Spring', 'Summer', 'Autumn', 'Winter'])
plt.title('Graph 3: Rented Bike Count Distribution by Season', fontsize=14, fontweight='bold')
plt.xlabel('Seasons', fontsize=12)
plt.ylabel('Rented Bike Count', fontsize=12)
plt.savefig('graphs/graph_3.png', dpi=300)
plt.close()

# --- GRAPH 4: Average Temperature vs Rented Bike Count ---
plt.figure(figsize=(10, 6))
temp_trend = df.groupby('Temperature(°C)')['Rented Bike Count'].mean().reset_index()
sns.lineplot(data=temp_trend, x='Temperature(°C)', y='Rented Bike Count', color='crimson', linewidth=2)
plt.title('Graph 4: Average Temperature vs Rented Bike Count', fontsize=14, fontweight='bold')
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Average Rented Bike Count', fontsize=12)
plt.savefig('graphs/graph_4.png', dpi=300)
plt.close()

# --- GRAPH 5: Correlation Heatmap of Numerical Features ---
plt.figure(figsize=(10, 8))
num_cols = df.select_dtypes(include=[np.number]).columns
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Graph 5: Correlation Heatmap of Numerical Features', fontsize=14, fontweight='bold')
plt.savefig('graphs/graph_5.png', dpi=300)
plt.close()

# --- GRAPH 6: Rented Bike Count on Holidays vs Non-Holidays ---
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Holiday', y='Rented Bike Count', palette='muted')
plt.title('Graph 6: Rented Bike Count on Holidays vs Non-Holidays', fontsize=14, fontweight='bold')
plt.xlabel('Holiday Status', fontsize=12)
plt.ylabel('Rented Bike Count', fontsize=12)
plt.savefig('graphs/graph_6.png', dpi=300)
plt.close()

# --- GRAPH 7: Hourly Bike Demand Pattern: Weekdays vs Weekends ---
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Hour', y='Rented Bike Count', hue='Is_Weekend', style='Is_Weekend', palette='Set1', marker='o')
plt.title('Graph 7: Hourly Bike Demand Pattern: Weekdays vs Weekends', fontsize=14, fontweight='bold')
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Average Rented Bike Count', fontsize=12)
plt.legend(title='Is Weekend', labels=['Weekday', 'Weekend'])
plt.savefig('graphs/graph_7.png', dpi=300)
plt.close()

# --- GRAPH 8: Average Monthly Trend of Rented Bikes ---
monthly_df = df.groupby('Month')['Rented Bike Count'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=monthly_df, x='Month', y='Rented Bike Count', marker='o', color='purple', linewidth=2.5, markersize=8)
plt.title('Graph 8: Average Monthly Trend of Rented Bikes', fontsize=14, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Average Rented Bike Count', fontsize=12)
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.savefig('graphs/graph_8.png', dpi=300)
plt.close()

# --- GRAPH 9: Average Bike Rentals Heatmap by Season and Hour ---
pivot_table = df.pivot_table(values='Rented Bike Count', index='Seasons', columns='Hour', aggfunc='mean')
pivot_table = pivot_table.reindex(['Spring', 'Summer', 'Autumn', 'Winter'])
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table, cmap='YlGnBu', annot=False, fmt='.0f', cbar_kws={'label': 'Mean Rented Bikes'})
plt.title('Graph 9: Average Bike Rentals Heatmap by Season and Hour', fontsize=14, fontweight='bold')
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Seasons', fontsize=12)
plt.savefig('graphs/graph_9.png', dpi=300)
plt.close()

# --- GRAPH 10: Average Rented Bike Count from Monday to Sunday (Bar Chart) ---
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_df = df.groupby('DayOfWeek')['Rented Bike Count'].mean().reindex(day_order).reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=daily_df, x='DayOfWeek', y='Rented Bike Count', palette='Blues_d', order=day_order)
plt.title('Graph 10: Average Rented Bike Count from Monday to Sunday', fontsize=14, fontweight='bold')
plt.xlabel('Day of the Week', fontsize=12)
plt.ylabel('Average Rented Bike Count', fontsize=12)
plt.savefig('graphs/graph_10.png', dpi=300)
plt.close()

# --- GRAPH 11: Average Rented Bike Count by Month (Bar Chart) ---
monthly_bar_df = df.groupby('Month')['Rented Bike Count'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=monthly_bar_df, x='Month', y='Rented Bike Count', palette='viridis')
plt.title('Graph 11: Average Rented Bike Count by Month', fontsize=14, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Average Rented Bike Count', fontsize=12)
plt.xticks(range(0, 12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.savefig('graphs/graph_11.png', dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
df['Rain_Status'] = df['Rainfall(mm)'].apply(lambda x: 'Rainy' if x > 0 else 'No Rain')
rain_df = df.groupby('Rain_Status')['Rented Bike Count'].mean().reset_index()
sns.barplot(data=rain_df, x='Rain_Status', y='Rented Bike Count', palette='coolwarm', order=['No Rain', 'Rainy'])
plt.title('Graph 12: Average Bike Rentals by Rain Condition', fontsize=14, fontweight='bold')
plt.xlabel('Rain Condition', fontsize=12)
plt.ylabel('Average Rented Bike Count', fontsize=12)
plt.savefig('graphs/graph_12.png', dpi=300)
plt.close()

# --- GRAPH 13: Average Bike Rentals by Holiday Status ---
plt.figure(figsize=(8, 6))
hol_df = df.groupby('Holiday')['Rented Bike Count'].mean().reset_index()
sns.barplot(data=hol_df, x='Holiday', y='Rented Bike Count', palette='muted')
plt.title('Graph 13: Average Bike Rentals by Holiday Status', fontsize=14, fontweight='bold')
plt.xlabel('Holiday Status', fontsize=12)
plt.ylabel('Average Rented Bike Count', fontsize=12)
plt.savefig('graphs/graph_13.png', dpi=300)
plt.close()

# --- GRAPH 14: Average Rented Bike Count Across Humidity Ranges ---
df['Humidity_Bin'] = pd.cut(df['Humidity(%)'], bins=[0, 20, 40, 60, 80, 100], labels=['0-20%', '21-40%', '41-60%', '61-80%', '81-100%'])
hum_df = df.groupby('Humidity_Bin', observed=True)['Rented Bike Count'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=hum_df, x='Humidity_Bin', y='Rented Bike Count', palette='crest')
plt.title('Graph 14: Average Rented Bike Count Across Humidity Levels', fontsize=14, fontweight='bold')
plt.xlabel('Humidity Range', fontsize=12)
plt.ylabel('Average Rented Bike Count', fontsize=12)
plt.savefig('graphs/graph_14.png', dpi=300)
plt.close()

# --- GRAPH 15: Average Rented Bike Count Across Temperature Ranges ---
df['Temp_Bin'] = pd.cut(df['Temperature(°C)'], bins=[-20, -10, 0, 10, 20, 30, 40], labels=['< -10°C', '-10 to 0°C', '0 to 10°C', '10 to 20°C', '20 to 30°C', '> 30°C'])
temp_bin_df = df.groupby('Temp_Bin', observed=True)['Rented Bike Count'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=temp_bin_df, x='Temp_Bin', y='Rented Bike Count', palette='magma')
plt.title('Graph 15: Average Rented Bike Count Across Temperature Ranges', fontsize=14, fontweight='bold')
plt.xlabel('Temperature Range', fontsize=12)
plt.ylabel('Average Rented Bike Count', fontsize=12)
plt.savefig('graphs/graph_15.png', dpi=300)
plt.close()

print("All refined graphs successfully generated and saved to the 'graphs/' folder!")