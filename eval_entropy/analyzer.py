import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy import stats
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import argparse
import ast
import math
from scipy.stats import chi2_contingency

def parse_logprobs(logprobs_str):
    """Parse the logprobs string and return list of probabilities."""
    try:
        # Convert string representation to list of dictionaries
        logprobs_list = ast.literal_eval(logprobs_str)
        
        # Extract logprobs and convert to probabilities
        probabilities = [2**float(item['logprob']) for item in logprobs_list]
        
        # Normalize probabilities to sum to 1
        total = sum(probabilities)
        normalized_probs = [p/total for p in probabilities]
        
        return normalized_probs
    except:
        return None

def calculate_entropy_metrics(probabilities):
    """Calculate entropy and varentropy from probabilities."""
    if not probabilities:
        return None, None
    
    # Calculate entropy: -Σ p_i * log2(p_i)
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    
    # Calculate varentropy: Σ p_i * (log2(p_i))^2 - (Σ p_i * log2(p_i))^2
    log_probs = [math.log2(p) for p in probabilities if p > 0]
    expected_log = sum(-p * lp for p, lp in zip(probabilities, log_probs))
    expected_square_log = sum(p * lp * lp for p, lp in zip(probabilities, log_probs))
    varentropy = expected_square_log - expected_log * expected_log
    
    return entropy, varentropy

def plot_contingency_table(contingency_table, metric, plots_dir):
    """Create a heatmap visualization of the contingency table."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlOrRd')
    plt.title(f'Contingency Table for {metric}')
    plt.xlabel('Correctly Answered')
    plt.ylabel('Metric Bins')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{metric}_contingency_table.png'))
    plt.close()

def perform_chi_squared_test(df, metric, stats_dir, plots_dir):
    """Perform chi-squared test of independence between metric and correctness."""
    # Discretize the metric into bins
    try:
        df['binned_metric'] = pd.qcut(df[metric], q=4, duplicates='drop')
    except ValueError:
        df['binned_metric'] = pd.cut(df[metric], bins=4)
    
    # Create contingency table
    contingency_table = pd.crosstab(df['binned_metric'], df['Correctly Answered'])
    
    # Perform chi-squared test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    # Save results
    chi_squared_file = os.path.join(stats_dir, f'chi_squared_test_{metric}.txt')
    with open(chi_squared_file, 'w') as f:
        f.write(f"Chi-Squared Test for {metric}:\n")
        f.write("="*50 + "\n")
        f.write(f"Chi-squared Statistic: {chi2:.4f}\n")
        f.write(f"Degrees of Freedom: {dof}\n")
        f.write(f"P-value: {p:.4e}\n\n")
        f.write("Contingency Table:\n")
        f.write(f"{contingency_table}\n\n")
        f.write("Expected Frequencies:\n")
        expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
        f.write(f"{expected_df}\n")
    
    # Plot the contingency table
    plot_contingency_table(contingency_table, metric, plots_dir)
    
    print(f"Chi-squared test for {metric} completed. Results saved to {chi_squared_file}")

def analyze_data(csv_file, output_dir, univariate_metrics=None, multivariate_pairs=None, 
                include_sqrt_varentropy=False, balance_classes=True):
    # Get base filename without extension for the output directory
    base_filename = os.path.splitext(os.path.basename(csv_file))[0]
    
    # Create specific output directory for this CSV
    csv_output_dir = os.path.join(output_dir, base_filename)
    plots_dir = os.path.join(csv_output_dir, 'plots')
    stats_dir = os.path.join(csv_output_dir, 'statistics')
    
    # Create all necessary directories
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    
    print(f"Saving results to: {csv_output_dir}")

    # Read the data
    df = pd.read_csv(csv_file)
    
    # Check if we have last_token_predictions and calculate entropy/varentropy
    if 'last_token_predictions' in df.columns:
        print("Found last_token_predictions, calculating entropy and varentropy from logprobs...")
        
        # Calculate entropy and varentropy from logprobs
        entropy_varentropy = [
            calculate_entropy_metrics(parse_logprobs(pred)) 
            for pred in df['last_token_predictions']
        ]
        
        # Add calculated metrics to dataframe
        df['entropy_from_logprobs'] = [ev[0] if ev else None for ev in entropy_varentropy]
        df['varentropy_from_logprobs'] = [ev[1] if ev else None for ev in entropy_varentropy]
        
        # Remove rows where calculation failed
        df = df.dropna(subset=['entropy_from_logprobs', 'varentropy_from_logprobs'])
        print(f"Calculated entropy and varentropy for {len(df)} samples")
        
        # Add these metrics to univariate_metrics if not specified
        if univariate_metrics is None:
            univariate_metrics = ['entropy_from_logprobs', 'varentropy_from_logprobs']
        else:
            univariate_metrics.extend(['entropy_from_logprobs', 'varentropy_from_logprobs'])
    
    # Drop rows where 'Correctly Answered' is NaN
    df = df.dropna(subset=['Correctly Answered'])
    
    # Convert 'Correctly Answered' to boolean
    df['Correctly Answered'] = df['Correctly Answered'].astype(bool)

    # Balance classes if requested
    if balance_classes:
        correct_samples = df[df['Correctly Answered']]
        incorrect_samples = df[~df['Correctly Answered']]
        
        # Get the size of the smaller class
        min_samples = min(len(correct_samples), len(incorrect_samples))
        
        # Randomly sample from both classes to get balanced dataset
        correct_samples = correct_samples.sample(n=min_samples, random_state=42)
        incorrect_samples = incorrect_samples.sample(n=min_samples, random_state=42)
        
        # Combine the balanced samples
        df = pd.concat([correct_samples, incorrect_samples])
        
        print(f"Balanced dataset: {min_samples} samples per class "
              f"(total {len(df)} samples)")

    # Add sqrt_varentropy if requested and varentropy exists
    if include_sqrt_varentropy and 'varentropy' in df.columns:
        df['sqrt_varentropy'] = np.sqrt(df['varentropy'])
        print("Added sqrt_varentropy metric")

    # Validate requested metrics exist in the dataframe
    available_metrics = [col for col in df.columns if col != 'Correctly Answered']
    print(f"Available metrics in dataset: {available_metrics}")

    if univariate_metrics:
        missing_metrics = [m for m in univariate_metrics if m not in df.columns]
        if missing_metrics:
            raise ValueError(f"Requested metrics not found in dataset: {missing_metrics}")
    else:
        univariate_metrics = available_metrics

    # Basic univariate analysis for each metric
    for metric in univariate_metrics:
        print(f"Analyzing metric: {metric}")
        
        # Basic statistics
        stats_file = os.path.join(stats_dir, f'{metric}_basic_stats.txt')
        with open(stats_file, 'w') as f:
            f.write(f"Basic statistics for {metric}:\n")
            f.write(f"Mean: {df[metric].mean():.4f}\n")
            f.write(f"Median: {df[metric].median():.4f}\n")
            f.write(f"Std Dev: {df[metric].std():.4f}\n")
            f.write(f"Min: {df[metric].min():.4f}\n")
            f.write(f"Max: {df[metric].max():.4f}\n\n")

        # Distribution plots
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df[df['Correctly Answered']], x=metric, bins=50, 
                    alpha=0.5, color='green', label='Correct')
        sns.histplot(data=df[~df['Correctly Answered']], x=metric, bins=50, 
                    alpha=0.5, color='red', label='Incorrect')
        plt.title(f'Distribution of {metric} by Correctness')
        plt.xlabel(metric)
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f'{metric}_distribution_by_correctness.png'))
        plt.close()

        # Perform chi-squared test
        perform_chi_squared_test(df, metric, stats_dir, plots_dir)

        # Univariate logistic regression
        X = df[[metric]]
        X = sm.add_constant(X)
        y = df['Correctly Answered']
        
        model = sm.Logit(y, X)
        results = model.fit()
        
        # Sklearn model for predictions
        sk_model = LogisticRegression()
        sk_model.fit(df[[metric]], y)
        y_pred = sk_model.predict(df[[metric]])
        y_pred_proba = sk_model.predict_proba(df[[metric]])[:, 1]
        
        with open(os.path.join(stats_dir, f'regression_results_{metric}_detailed.txt'), 'w') as f:
            f.write(f"Detailed Logistic Regression Results for {metric}:\n")
            f.write("="*50 + "\n")
            f.write(results.summary().as_text())
            f.write("\n\nAdditional Metrics:\n")
            f.write(f"ROC AUC Score: {roc_auc_score(y, y_pred_proba):.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(classification_report(y, y_pred))

    # Multivariate analysis
    if multivariate_pairs:
        for metric_pair in multivariate_pairs:
            if not all(metric in df.columns for metric in metric_pair):
                print(f"Skipping pair {metric_pair} - not all metrics available")
                continue
                
            print(f"Analyzing metric pair: {metric_pair}")
            
            X_combined = df[list(metric_pair)]
            X_combined = sm.add_constant(X_combined)
            model_combined = sm.Logit(y, X_combined)
            results_combined = model_combined.fit()
            
            sk_model_combined = LogisticRegression()
            sk_model_combined.fit(df[list(metric_pair)], y)
            y_pred_combined = sk_model_combined.predict(df[list(metric_pair)])
            y_pred_proba_combined = sk_model_combined.predict_proba(df[list(metric_pair)])[:, 1]
            
            pair_name = '_'.join(metric_pair)
            with open(os.path.join(stats_dir, f'regression_results_{pair_name}_detailed.txt'), 'w') as f:
                f.write(f"Detailed Logistic Regression Results for {pair_name}:\n")
                f.write("="*50 + "\n")
                f.write(results_combined.summary().as_text())
                f.write("\n\nAdditional Metrics:\n")
                f.write(f"ROC AUC Score: {roc_auc_score(y, y_pred_proba_combined):.4f}\n")
                f.write("\nClassification Report:\n")
                f.write(classification_report(y, y_pred_combined))

    # Create correlation matrix plot for all analyzed metrics
    analyzed_metrics = list(set(univariate_metrics + [m for pair in (multivariate_pairs or []) for m in pair]))
    if len(analyzed_metrics) > 1:
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[analyzed_metrics + ['Correctly Answered']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'))
        plt.close()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze entropy data from CSV files')
    parser.add_argument('--input-csv', help='CSV file to process')
    parser.add_argument('--output-dir', default='output', help='Output directory for analysis results')
    parser.add_argument('--univariate-metrics', nargs='+', help='Metrics to analyze individually')
    parser.add_argument('--multivariate-pairs', nargs='+', action='append', 
                       help='Pairs of metrics to analyze together. Use multiple times for multiple pairs.')
    parser.add_argument('--include-sqrt-varentropy', action='store_true',
                       help='Calculate and include sqrt_varentropy metric')
    parser.add_argument('--no-balance-classes', action='store_true',
                       help='Disable automatic class balancing')
    args = parser.parse_args()

    # Process the input file
    if args.input_csv:
        if os.path.exists(args.input_csv):
            print(f"Processing {args.input_csv}...")
            
            # Convert multivariate pairs from flat list to tuple pairs
            multivariate_pairs = None
            if args.multivariate_pairs:
                multivariate_pairs = []
                for pair in args.multivariate_pairs:
                    if len(pair) >= 2:
                        for i in range(0, len(pair), 2):
                            if i + 1 < len(pair):
                                multivariate_pairs.append((pair[i], pair[i+1]))
            
            analyze_data(
                args.input_csv, 
                args.output_dir,
                univariate_metrics=args.univariate_metrics,
                multivariate_pairs=multivariate_pairs,
                include_sqrt_varentropy=args.include_sqrt_varentropy,
                balance_classes=not args.no_balance_classes
            )
            print(f"Analysis complete for {args.input_csv}")
        else:
            print(f"Error: File {args.input_csv} not found")
    else:
        # Default behavior if no input CSV specified
        default_files = [
            'entropy_text_correctly_answered.csv',
            'entropy_varentropy_text_correctly_answered.csv'
        ]
        for file in default_files:
            if os.path.exists(file):
                print(f"Processing {file}...")
                analyze_data(file, args.output_dir)
                print(f"Analysis complete for {file}")

if __name__ == "__main__":
    main()


