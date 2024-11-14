import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
import os

def main():
    # ------------------------------
    # Configuration
    # ------------------------------
    
    # Path to the data file
    data_file = 'entropy_text_correctly_answered.csv'
    
    # Directory to save the plots and statistics
    plots_dir = os.path.join('output', 'plots')
    stats_dir = os.path.join('output', 'statistics')
    
    # Ensure the output directories exist
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    
    # ------------------------------
    # 1. Load and Inspect the Data
    # ------------------------------
    
    # Load the data
    try:
        df = pd.read_csv(data_file)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file '{data_file}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading '{data_file}': {e}")
        return
    
    print("\nFirst few rows of the data:")
    print(df.head())
    
    # ------------------------------
    # 2. Data Preprocessing
    # ------------------------------
    
    # Convert 'Correctly Answered' to boolean (1 for True, 0 for False)
    df['Correctly Answered'] = df['Correctly Answered'].astype(str).str.strip().str.lower()
    df['Correctly Answered'] = df['Correctly Answered'].map({'true': 1, 'false': 0})
    
    # Drop rows with missing values in 'Correctly Answered' or 'entropy'
    initial_row_count = len(df)
    df = df.dropna(subset=['Correctly Answered', 'entropy'])
    final_row_count = len(df)
    print(f"\nDropped {initial_row_count - final_row_count} rows due to missing values.")
    
    # Convert 'entropy' to numeric
    df['entropy'] = pd.to_numeric(df['entropy'], errors='coerce')
    
    # Drop rows with NaN entropy after conversion
    df = df.dropna(subset=['entropy'])
    
    # ------------------------------
    # 3. Perform Logistic Regression
    # ------------------------------
    
    # Define the dependent and independent variables
    X = df['entropy']
    y = df['Correctly Answered']
    
    # Add a constant term for the intercept
    X_const = sm.add_constant(X)
    
    # Fit the logistic regression model
    try:
        model = sm.Logit(y, X_const)
        result = model.fit(disp=False)  # disp=False suppresses the fitting output
        print("\nLogistic Regression Model fitted successfully.")
    except Exception as e:
        print(f"An error occurred during logistic regression fitting: {e}")
        return
    
    # Print the summary of the model
    print("\nLogistic Regression Results:")
    print(result.summary())
    
    # ------------------------------
    # 4. Plotting
    # ------------------------------
    
    # 4.1. Scatter Plot with Logistic Regression Curve
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of the raw data
    sns.scatterplot(x='entropy', y='Correctly Answered', data=df, alpha=0.5, label='Data Points')
    
    # Sort the entropy values for plotting the regression curve
    X_sorted = np.sort(X)
    X_const_sorted = sm.add_constant(X_sorted)
    
    # Predict probabilities using the logistic regression model
    y_pred = result.predict(X_const_sorted)
    
    # Plot the regression curve
    plt.plot(X_sorted, y_pred, color='red', linewidth=2, label='Logistic Regression')
    
    # Labels and title
    plt.xlabel('Entropy')
    plt.ylabel('Probability of Correct Answer')
    plt.title('Entropy vs. Probability of Correct Answer')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    scatter_reg_plot_path = os.path.join(plots_dir, 'entropy_vs_correctness.png')
    plt.savefig(scatter_reg_plot_path, dpi=300, bbox_inches='tight')
    print(f"\nScatter plot with regression curve saved as '{scatter_reg_plot_path}'.")
    
    # Optional: Display the plot
    plt.show()
    
    # 4.2. Histogram of Entropy for Correct and Incorrect Answers
    plt.figure(figsize=(10, 6))
    
    # Correct answers
    sns.histplot(df[df['Correctly Answered'] == 1]['entropy'], 
                 color='green', 
                 label='Correct', 
                 kde=True, 
                 stat='density', 
                 bins=30, 
                 alpha=0.6)
    
    # Incorrect answers
    sns.histplot(df[df['Correctly Answered'] == 0]['entropy'], 
                 color='red', 
                 label='Incorrect', 
                 kde=True, 
                 stat='density', 
                 bins=30, 
                 alpha=0.6)
    
    # Labels and title
    plt.xlabel('Entropy')
    plt.ylabel('Density')
    plt.title('Distribution of Entropy for Correct and Incorrect Answers')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    entropy_dist_plot_path = os.path.join(plots_dir, 'entropy_distribution.png')
    plt.savefig(entropy_dist_plot_path, dpi=300, bbox_inches='tight')
    print(f"Entropy distribution histogram saved as '{entropy_dist_plot_path}'.")
    
    # Optional: Display the plot
    plt.show()
    
    # 4.3. Additional Plot: Bar Chart of Frequency of Correct Responses
    plt.figure(figsize=(8, 6))
    
    # Frequency counts
    frequency = df['Correctly Answered'].value_counts().sort_index()
    sns.barplot(x=frequency.index.map({0: 'Incorrect', 1: 'Correct'}), y=frequency.values, palette=['red', 'green'])
    
    # Labels and title
    plt.xlabel('Response Correctness')
    plt.ylabel('Frequency')
    plt.title('Frequency of Correct and Incorrect Responses')
    plt.grid(axis='y')
    
    # Save the plot
    frequency_plot_path = os.path.join(plots_dir, 'response_frequency.png')
    plt.savefig(frequency_plot_path, dpi=300, bbox_inches='tight')
    print(f"Response frequency bar chart saved as '{frequency_plot_path}'.")
    
    # Optional: Display the plot
    plt.show()
    
    # ------------------------------
    # 5. Statistical Analyses
    # ------------------------------
    
    # 5.1. Frequency and Percentage of Correct Responses
    frequency = df['Correctly Answered'].value_counts().sort_index()
    percentage = df['Correctly Answered'].value_counts(normalize=True).sort_index() * 100
    
    print("\nFrequency of Correct Responses:")
    print(frequency.rename({0: 'Incorrect', 1: 'Correct'}))
    
    print("\nPercentage of Correct Responses:")
    print(percentage.rename({0: 'Incorrect', 1: 'Correct'}).round(2))
    
    # 5.2. Distribution of Entropy for Correct and Incorrect Answers
    # (Already visualized in the histogram above)
    
    # 5.3. One-way ANOVA Test
    # Separate entropy values for correct and incorrect answers
    entropy_correct = df[df['Correctly Answered'] == 1]['entropy']
    entropy_incorrect = df[df['Correctly Answered'] == 0]['entropy']
    
    # Perform one-way ANOVA
    f_statistic, p_value = stats.f_oneway(entropy_correct, entropy_incorrect)
    
    print("\nOne-way ANOVA Results:")
    print(f"F-statistic: {f_statistic:.4f}")
    print(f"p-value: {p_value:.4e}")
    
    # Add descriptive statistics
    print("\nDescriptive Statistics:")
    print("\nCorrect Answers:")
    print(f"Mean entropy: {entropy_correct.mean():.4f}")
    print(f"Standard deviation: {entropy_correct.std():.4f}")
    print(f"Count: {len(entropy_correct)}")
    
    print("\nIncorrect Answers:")
    print(f"Mean entropy: {entropy_incorrect.mean():.4f}")
    print(f"Standard deviation: {entropy_incorrect.std():.4f}")
    print(f"Count: {len(entropy_incorrect)}")
    
    # Save ANOVA results to a file
    anova_path = os.path.join(stats_dir, 'anova_results.txt')
    with open(anova_path, 'w') as f:
        f.write("One-way ANOVA Results\n")
        f.write("-" * 20 + "\n\n")
        f.write(f"F-statistic: {f_statistic:.4f}\n")
        f.write(f"p-value: {p_value:.4e}\n\n")
        f.write("Descriptive Statistics\n")
        f.write("-" * 20 + "\n\n")
        f.write("Correct Answers:\n")
        f.write(f"Mean entropy: {entropy_correct.mean():.4f}\n")
        f.write(f"Standard deviation: {entropy_correct.std():.4f}\n")
        f.write(f"Count: {len(entropy_correct)}\n\n")
        f.write("Incorrect Answers:\n")
        f.write(f"Mean entropy: {entropy_incorrect.mean():.4f}\n")
        f.write(f"Standard deviation: {entropy_incorrect.std():.4f}\n")
        f.write(f"Count: {len(entropy_incorrect)}\n")
    
    print(f"\nANOVA results saved to '{anova_path}'.")
    
    # Optionally, save the statistical summaries to text files
    frequency_path = os.path.join(stats_dir, 'frequency_of_correct_responses.txt')
    percentage_path = os.path.join(stats_dir, 'percentage_of_correct_responses.txt')
    
    frequency.rename({0: 'Incorrect', 1: 'Correct'}).to_csv(frequency_path, sep='\t')
    percentage.rename({0: 'Incorrect', 1: 'Correct'}).round(2).to_csv(percentage_path, sep='\t')
    
    print(f"\nFrequency data saved to '{frequency_path}'.")
    print(f"Percentage data saved to '{percentage_path}'.")
    
    # ------------------------------
    # 6. Conclusion
    # ------------------------------
    
    print("\nAll plots have been saved successfully in the 'output/plots' directory.")
    print("Statistical summaries have been saved in the 'output/statistics' directory.")

if __name__ == "__main__":
    main()
