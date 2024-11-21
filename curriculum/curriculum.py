import os
import sys
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
import gc
import json
from pathlib import Path
from datatrove.pipeline.readers import ParquetReader

# Device configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")

torch.set_float32_matmul_precision("high")
print(f"Using device: {device}")

# Load Llama tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
llama_tokenizer.pad_token = llama_tokenizer.eos_token

class DatasetBatchIterator:
    """Iterator for processing documents from ParquetReader in batches."""
    def __init__(self, dataset_path: str, batch_size: int = 32, seq_length: int = 512, limit: int = None):
        self.reader = ParquetReader(dataset_path, limit=limit)
        self.batch_size = batch_size
        self.seq_length = seq_length
        
    def __iter__(self):
        current_batch = []
        
        for document in self.reader():
            # Tokenize the document text
            encoded = llama_tokenizer(
                document.text,
                truncation=True,
                padding='max_length',
                max_length=self.seq_length,
                return_tensors='pt'
            )
            
            current_batch.append(encoded)
            
            if len(current_batch) >= self.batch_size:
                # Combine batch tensors
                batch_input_ids = torch.cat([item.input_ids for item in current_batch], dim=0)
                batch_attention_mask = torch.cat([item.attention_mask for item in current_batch], dim=0)
                
                yield batch_input_ids, batch_attention_mask
                current_batch = []
        
        # Yield remaining batch
        if current_batch:
            batch_input_ids = torch.cat([item.input_ids for item in current_batch], dim=0)
            batch_attention_mask = torch.cat([item.attention_mask for item in current_batch], dim=0)
            yield batch_input_ids, batch_attention_mask

class FluxEntropy:
    def __init__(self, model, num_buckets: int = 6, seed: int = 1337):
        self.model = model
        self.num_buckets = num_buckets
        torch.manual_seed(seed)
        
        self.stats = {
            'sequences_processed': 0,
            'entropy_stats': {
                'min': float('inf'),
                'max': float('-inf'),
                'mean': 0,
                'std': 0,
                'sequences_per_bucket': [0] * num_buckets
            }
        }
        
        self.bucket_boundaries = None

    def compute_entropy(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute per-token entropy and then average."""
        with torch.no_grad(), torch.cuda.amp.autocast():
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            batch_size, seq_len = input_ids.shape

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            log_probs = torch.log_softmax(logits, dim=-1)
            entropy_tensor = -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)

            masked_entropy = entropy_tensor * attention_mask
            seq_lengths = attention_mask.sum(dim=1)
            avg_entropy = masked_entropy.sum(dim=1) / seq_lengths

            return avg_entropy.cpu()

    def process_dataset(self, dataset_path: str, output_dir: str, limit: int = None):
        """Process the dataset and distribute sequences to entropy buckets."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nProcessing dataset from {dataset_path}")

        all_entropies = []
        sequence_map = []

        iterator = DatasetBatchIterator(
            dataset_path=dataset_path,
            batch_size=26,
            limit=limit
        )

        for batch_idx, (batch_ids, batch_mask) in enumerate(tqdm(iterator)):
            try:
                batch_entropy = self.compute_entropy(batch_ids, batch_mask)
                
                all_entropies.extend(batch_entropy.tolist())
                for i in range(len(batch_ids)):
                    sequence_map.append((batch_idx * iterator.batch_size + i, batch_ids[i].numpy()))

                if batch_idx == 0:
                    print("\nSample entropy values from first batch:")
                    for i, entropy in enumerate(batch_entropy.tolist()):
                        print(f"Sequence {i}: {entropy:.4f}")

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                continue

            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        if not all_entropies:
            print("No sequences processed successfully")
            return

        # Compute statistics
        entropy_array = np.array(all_entropies)
        entropy_std = np.std(entropy_array)
        entropy_mean = np.mean(entropy_array)

        print("\nEntropy Statistics:")
        print(f"Mean: {entropy_mean:.4f}")
        print(f"Std: {entropy_std:.4f}")
        print(f"Min: {np.min(entropy_array):.4f}")
        print(f"Max: {np.max(entropy_array):.4f}")
        print(f"\nPercentiles:")
        percentiles = [0, 10, 25, 50, 75, 90, 100]
        for p in percentiles:
            print(f"{p}th percentile: {np.percentile(entropy_array, p):.4f}")

        # Compute or use bucket boundaries
        if limit is None:
            print("\nComputing bucket boundaries...")
            # Use non-uniform quantiles for better distribution
            quantiles = [10, 25, 50, 75, 90]  # Changed from uniform to percentile-based
            self.bucket_boundaries = np.percentile(all_entropies, quantiles)

            # Save boundaries and detailed statistics
            with open(os.path.join(output_dir, 'bucket_boundaries.json'), 'w') as f:
                json.dump({
                    'boundaries': self.bucket_boundaries.tolist(),
                    'stats': {
                        'mean': float(entropy_mean),
                        'std': float(entropy_std),
                        'min': float(np.min(entropy_array)),
                        'max': float(np.max(entropy_array)),
                        'percentiles': {
                            str(p): float(np.percentile(entropy_array, p))
                            for p in percentiles
                        }
                    },
                    'quantiles_used': quantiles
                }, f, indent=2)

        # Distribute sequences to buckets
        bucket_sequences = [[] for _ in range(self.num_buckets)]
        for (seq_idx, sequence), entropy in zip(sequence_map, all_entropies):
            bucket_idx = np.searchsorted(self.bucket_boundaries, entropy)
            if bucket_idx >= self.num_buckets:
                bucket_idx = self.num_buckets - 1
            bucket_sequences[bucket_idx].append((entropy, sequence))
            self.stats['entropy_stats']['sequences_per_bucket'][bucket_idx] += 1

        # Print bucket distribution
        print("\nBucket Distribution:")
        for i, sequences in enumerate(bucket_sequences):
            count = len(sequences)
            percentage = (count / len(all_entropies)) * 100
            print(f"Bucket {i+1}: {count} sequences ({percentage:.2f}%)")

        # Update stats
        self.stats['sequences_processed'] += len(all_entropies)
        self.stats['entropy_stats'].update({
            'min': float(np.min(entropy_array)),
            'max': float(np.max(entropy_array)),
            'mean': float(entropy_mean),
            'std': float(entropy_std)
        })

def process_test():
    """Process a sample of the dataset for testing."""
    print("Initializing model...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    processor = FluxEntropy(model)
    
    # Use the 100BT sample dataset
    dataset_path = "hf://datasets/HuggingFaceFW/fineweb/sample/100BT"
    output_dir = "./entropy_buckets"
    
    # Process with limit
    processor.process_dataset(
        dataset_path=dataset_path,
        output_dir=output_dir,
        limit=1000  # Process 1000 documents for testing
    )

if __name__ == "__main__":
    process_test()
