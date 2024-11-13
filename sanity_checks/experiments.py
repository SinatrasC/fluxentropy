import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import scipy.stats as stats
from statsmodels.formula.api import ols
import statsmodels.api as sm

def main():
    # Define directories
    data_dir = 'experiment_data'
    compiled_csv = 'compiled_results.csv'
    output_dir = 'plots'  # Updated specific output directory

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Read the compiled CSV file
    if not os.path.exists(compiled_csv):
        print(f"Compiled CSV file '{compiled_csv}' not found. Please run the compilation script first.")
        return

    df = pd.read_csv(compiled_csv)

    # Extract model names from the columns
    columns = df.columns.tolist()
    model_names = set()
    for col in columns:
        if '_Characteristic_Value' in col:
            model_name = col.replace('_Characteristic_Value', '')
            model_names.add(model_name)

    # After extracting model names, add this name processing
    def simplify_model_name(name):
        parts = name.split('_')[:2]  # Take first two parts
        return '_'.join(parts)
    
    # Update model names
    model_names = sorted(model_names)
    model_name_mapping = {name: simplify_model_name(name) for name in model_names}

    # Step 2: Construct entropy vectors per model
    entropy_vectors = {}
    rank_vectors = {}
    for model in model_names:
        entropy_col = f'{model}_Characteristic_Value'
        rank_col = f'{model}_Rank'
        entropy_vectors[model] = df[entropy_col].values
        rank_vectors[model] = df[rank_col].values

    # Create a DataFrame for entropy values
    entropy_df = pd.DataFrame(entropy_vectors)

    # Step 3: Compute pairwise Euclidean distances between models
    distances = euclidean_distances(entropy_df.T)

    # Plot the distance matrix
    plt.figure(figsize=(8, 6))
    simplified_names = [model_name_mapping[name] for name in model_names]
    sns.heatmap(distances, annot=True, xticklabels=simplified_names, yticklabels=simplified_names, cmap='viridis')
    plt.title('Pairwise Euclidean Distance Between Models')
    plt.tight_layout()
    distance_matrix_path = os.path.join(output_dir, 'distance_matrix.png')
    plt.savefig(distance_matrix_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to free memory
    print(f"Distance matrix plot saved to {distance_matrix_path}")

    # Step 4: Perform PCA on the entropy vectors
    pca = PCA(n_components=2)
    entropy_vectors_matrix = entropy_df.values.T  # Models as rows
    pca_result = pca.fit_transform(entropy_vectors_matrix)

    # Plot the models in 2D space
    plt.figure(figsize=(8, 6))
    for i, model in enumerate(model_names):
        simplified_name = model_name_mapping[model]
        plt.scatter(pca_result[i, 0], pca_result[i, 1], label=simplified_name)
        plt.text(pca_result[i, 0] + 0.01, pca_result[i, 1] + 0.01, simplified_name)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Models Based on Entropy Vectors')
    plt.legend(prop={'size': 8})
    plt.tight_layout()
    pca_models_path = os.path.join(output_dir, 'pca_models.png')
    plt.savefig(pca_models_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot
    print(f"PCA models plot saved to {pca_models_path}")

    # Plot explained variance by the two principal components
    explained_variance = pca.explained_variance_ratio_
    plt.figure(figsize=(6, 4))
    plt.bar(['PC1', 'PC2'], explained_variance * 100, color=['skyblue', 'salmon'])
    plt.ylabel('Explained Variance (%)')
    plt.title('Explained Variance by Principal Components')
    plt.tight_layout()
    explained_variance_path = os.path.join(output_dir, 'explained_variance.png')
    plt.savefig(explained_variance_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot
    print(f"Explained variance plot saved to {explained_variance_path}")

    # Step 5: Prepare data for scatter plots
    texts = df['Text']
    combined_data = pd.DataFrame()

    for model in model_names:
        temp_df = pd.DataFrame({
            'Text': texts,
            'Model': model_name_mapping[model],  # Use simplified name
            'Rank': df[f'{model}_Rank'],
            'Entropy': df[f'{model}_Characteristic_Value']
        })
        combined_data = pd.concat([combined_data, temp_df], ignore_index=True)

    # Scatter plot for ranks
    plt.figure(figsize=(12, 8))
    combined_data['Index'] = combined_data.groupby('Model').cumcount()
    sns.scatterplot(data=combined_data, x='Index', y='Rank', hue='Model', palette='tab10', alpha=0.7)
    plt.xticks(visible=False)
    plt.title('Ranks Assigned by Different Models')
    plt.ylabel('Rank')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 8})
    plt.tight_layout()
    ranks_scatter_path = os.path.join(output_dir, 'ranks_scatter.png')
    plt.savefig(ranks_scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Ranks scatter plot saved to {ranks_scatter_path}")

    # Scatter plot for entropies
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=combined_data, x='Index', y='Entropy', hue='Model', palette='tab10', alpha=0.7)
    plt.xticks(visible=False)
    plt.title('Entropies Assigned by Different Models')
    plt.ylabel('Entropy')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 8})
    plt.tight_layout()
    entropies_scatter_path = os.path.join(output_dir, 'entropies_scatter.png')
    plt.savefig(entropies_scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Entropies scatter plot saved to {entropies_scatter_path}")

    # Calculate correlation coefficients and add line fits for ranks
    for model in model_name_mapping.values():  # Iterate over simplified model names
        model_data = combined_data[combined_data['Model'] == model]
        if not model_data.empty:
            slope, intercept, r_value, p_value, std_err = stats.linregress(model_data['Index'], model_data['Rank'])
            plt.plot(model_data['Index'], intercept + slope * model_data['Index'], label=f'{model} Fit (r={r_value:.2f})')
        else:
            print(f"No data available for model: {model} to perform linear regression.")
    
    # Calculate correlation coefficients and add line fits for entropies
    for model in model_name_mapping.values():  # Iterate over simplified model names
        model_data = combined_data[combined_data['Model'] == model]
        if not model_data.empty:
            slope, intercept, r_value, p_value, std_err = stats.linregress(model_data['Index'], model_data['Entropy'])
            plt.plot(model_data['Index'], intercept + slope * model_data['Index'], label=f'{model} Fit (r={r_value:.2f})')
        else:
            print(f"No data available for model: {model} to perform linear regression.")

    # After plotting scatter plots, add statistical tests
    # Example using ANOVA to compare correlations

    # Prepare data for ANOVA on ranks
    aov_rank = ols('Rank ~ C(Model)', data=combined_data).fit()
    table_rank = sm.stats.anova_lm(aov_rank, typ=2)
    print("ANOVA results for Rank by Model:")
    print(table_rank)

    # Prepare data for ANOVA on entropies
    aov_entropy = ols('Entropy ~ C(Model)', data=combined_data).fit()
    table_entropy = sm.stats.anova_lm(aov_entropy, typ=2)
    print("ANOVA results for Entropy by Model:")
    print(table_entropy)

    # Enhance PCA plot aesthetics
    plt.figure(figsize=(10, 8))
    for i, model in enumerate(model_names):
        simplified_name = model_name_mapping[model]
        plt.scatter(pca_result[i, 0], pca_result[i, 1], label=simplified_name, s=100)  # Increased marker size
        plt.text(pca_result[i, 0] + 0.02, pca_result[i, 1] + 0.02, simplified_name, fontsize=9, 
                 horizontalalignment='left', verticalalignment='bottom')
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.title('PCA of Models Based on Entropy Vectors', fontsize=14)
    plt.legend(prop={'size': 10}, loc='best', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    pca_models_path = os.path.join(output_dir, 'pca_models.png')
    plt.savefig(pca_models_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot
    print(f"PCA models plot saved to {pca_models_path}")

if __name__ == '__main__':
    main()
