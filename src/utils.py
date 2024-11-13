# utils.py
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def compute_entropy(logits):
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # Shape: (batch_size, seq_len)
    return entropy

def compute_varentropy(logits, entropy=None):
    if entropy is None:
        entropy = compute_entropy(logits)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    varentropy = torch.sum(probs * (log_probs + entropy.unsqueeze(-1)) ** 2, dim=-1)
    return varentropy

def permute_dataset(dataset, characteristics, descending=False):
    characteristic_np = characteristics.cpu().numpy()
    sorted_indices = np.argsort(characteristic_np)
    if descending:
        sorted_indices = sorted_indices[::-1]
    permuted_dataset = [dataset[idx] for idx in sorted_indices]
    sorted_characteristics = characteristic_np[sorted_indices]
    return permuted_dataset, sorted_characteristics

def visualize_results(results, config, title=None, height=800):
    mechanism = config.get("mechanism", "per_token")
    compute_entropy = config.get("compute_entropy", False)
    compute_varentropy = config.get("compute_varentropy", False)

    if mechanism == "per_string":
        fig = go.Figure()

        if compute_entropy:
            fig.add_trace(
                go.Bar(
                    x=results['input_strings'],
                    y=results['entropy'],
                    text=results['entropy'],
                    textposition='auto',
                    name='Entropy'
                )
            )

        if compute_varentropy:
            fig.add_trace(
                go.Bar(
                    x=results['input_strings'],
                    y=results['varentropy'],
                    text=results['varentropy'],
                    textposition='auto',
                    name='Varentropy'
                )
            )

        fig.update_layout(
            title=title or 'Entropy and Varentropy Analysis (Full String)',
            xaxis_title='Input Strings',
            yaxis_title='Value',
            barmode='group',
            height=height
        )

    elif mechanism == "per_token":
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Entropy Over Tokens', 'Varentropy Over Tokens', 'Token-wise Analysis'),
            vertical_spacing=0.1,
            row_heights=[0.35, 0.35, 0.3]
        )

        for batch_idx, input_string in enumerate(results['input_strings']):
            mask = results['attention_mask'][batch_idx]
            seq_len = mask.sum().item()

            tokens = results['tokens'][batch_idx][:seq_len]
            positions = np.arange(seq_len)

            if compute_entropy:
                entropy_values = results['entropy'][batch_idx][:seq_len].numpy()
                fig.add_trace(
                    go.Scatter(
                        x=positions,
                        y=entropy_values,
                        mode='lines+markers',
                        name=f'Entropy (String {batch_idx + 1})',
                        hovertemplate='Position: %{x}<br>Entropy: %{y:.3f}<extra></extra>'
                    ),
                    row=1, col=1
                )

            if compute_varentropy:
                varentropy_values = results['varentropy'][batch_idx][:seq_len].numpy()
                fig.add_trace(
                    go.Scatter(
                        x=positions,
                        y=varentropy_values,
                        mode='lines+markers',
                        name=f'Varentropy (String {batch_idx + 1})',
                        hovertemplate='Position: %{x}<br>Varentropy: %{y:.3f}<extra></extra>'
                    ),
                    row=2, col=1
                )

            # Token-wise heatmap for entropy and varentropy
            heatmap_z = []
            heatmap_y = []
            if compute_entropy:
                heatmap_z.append(results["entropy"][batch_idx][:seq_len].numpy())
                heatmap_y.append('Entropy')
            if compute_varentropy:
                heatmap_z.append(results["varentropy"][batch_idx][:seq_len].numpy())
                heatmap_y.append('Varentropy')

            if heatmap_z:
                fig.add_trace(
                    go.Heatmap(
                        z=heatmap_z,
                        x=tokens,
                        y=heatmap_y,
                        colorscale='Viridis',
                        showscale=True,
                        hoverongaps=False,
                        hovertemplate='Token: %{x}<br>Metric: %{y}<br>Value: %{z:.3f}<extra></extra>'
                    ),
                    row=3, col=1
                )

        fig.update_layout(
            height=height,
            showlegend=True,
            title=title or 'Entropy and Varentropy Analysis',
            hovermode='closest'
        )

        # Update axes labels
        fig.update_xaxes(title_text='Token Position', row=1, col=1)
        fig.update_xaxes(title_text='Token Position', row=2, col=1)
        fig.update_xaxes(title_text='Tokens', row=3, col=1)

        if compute_entropy:
            fig.update_yaxes(title_text='Entropy', row=1, col=1)
        if compute_varentropy:
            fig.update_yaxes(title_text='Varentropy', row=2, col=1)

    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

    return fig
