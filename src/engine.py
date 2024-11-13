# engine.py
import torch
import utils

class Engine:
    def __init__(self, model, tokenizer, device, dtype=torch.bfloat16):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype

    def compute_characteristics(self, input_strings, config, max_length=512):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        encodings = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        input_ids = encodings["input_ids"].to(self.device)
        padding_mask = encodings["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=padding_mask)
            logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

        results = {
            "input_strings": input_strings,
            "tokens": [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids],
            "attention_mask": padding_mask.cpu()
        }

        mechanism = config.get("mechanism", "per_token")
        compute_entropy_flag = config.get("compute_entropy", False)
        compute_varentropy_flag = config.get("compute_varentropy", False)

        if compute_entropy_flag:
            entropy = utils.compute_entropy(logits)
            results["entropy"] = entropy.cpu()

        if compute_varentropy_flag:
            varentropy = utils.compute_varentropy(logits, entropy)
            results["varentropy"] = varentropy.cpu()

        return results

    def permute_dataset(self, dataset, config, sort_by, descending=False, max_length=512):
        results = self.compute_characteristics(
            input_strings=dataset,
            config=config,
            max_length=max_length
        )

        mechanism = config.get("mechanism", "per_token")
        compute_entropy_flag = config.get("compute_entropy", False)
        compute_varentropy_flag = config.get("compute_varentropy", False)

        if sort_by not in results:
            if mechanism == "per_token":
                padding_mask = results["attention_mask"]
                if sort_by == "entropy_token_avg" and compute_entropy_flag:
                    entropy = results["entropy"]
                    masked_entropy = torch.sum(entropy * padding_mask, dim=1) / torch.sum(padding_mask, dim=1)
                    characteristic = masked_entropy
                elif sort_by == "entropy_token_sum" and compute_entropy_flag:
                    entropy = results["entropy"]
                    masked_entropy = torch.sum(entropy * padding_mask, dim=1)
                    characteristic = masked_entropy
                elif sort_by == "varentropy_token_avg" and compute_varentropy_flag:
                    varentropy = results["varentropy"]
                    masked_varentropy = torch.sum(varentropy * padding_mask, dim=1) / torch.sum(padding_mask, dim=1)
                    characteristic = masked_varentropy
                elif sort_by == "varentropy_token_sum" and compute_varentropy_flag:
                    varentropy = results["varentropy"]
                    masked_varentropy = torch.sum(varentropy * padding_mask, dim=1)
                    characteristic = masked_varentropy
                else:
                    raise ValueError(f"Unknown sort_by option: {sort_by}")
            else:
                raise ValueError(f"sort_by '{sort_by}' not found in results and mechanism is '{mechanism}'")
        else:
            characteristic = results[sort_by]

        permuted_dataset, sorted_characteristics = utils.permute_dataset(
            dataset, characteristic, descending
        )
        return permuted_dataset, sorted_characteristics
