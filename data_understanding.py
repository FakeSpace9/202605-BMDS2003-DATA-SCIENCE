import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_data_understanding():
    # Load dataset
    # Note: SeoulBikeData often uses 'unicode_escape' or 'latin1' encoding due to special characters
    file_path = 'SeoulBikeData.csv'
    df = pd.read_csv(file_path, encoding='unicode_escape')

    print("--- Data Understanding ---")
    print("\n1. Dataset Info:")
    df.info()

    print("\n2. Summary Statistics:")
    summary_stats = df.describe(include='all')
    print(summary_stats)
    
    # Save summary stats to CSV for easy inclusion in report
    summary_stats.to_csv("summary_statistics.csv")
    print("\nSummary statistics saved to 'summary_statistics.csv'")

    # Visualizations
    os.makedirs('plots', exist_ok=True)

    # Distribution of the target variable
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Rented Bike Count'], bins=30, kde=True)
    plt.title('Distribution of Rented Bike Count')
    plt.savefig('plots/target_distribution.png')
    plt.close()

    # Correlation Matrix (Numerical only)
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig('plots/correlation_heatmap.png')
    plt.close()
    
    print("\nGraphs saved in the 'plots' folder for your report.")

if __name__ == "__main__":
    run_data_understanding()