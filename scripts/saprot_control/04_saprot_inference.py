#!/usr/bin/env python3
"""
Step 4: SaProt inference — generate 1280-dim embeddings.
Model: westlake-repl/SaProt_650M_AF2
Method: model.esm encoder → last_hidden_state → mean pool (attention_mask)
Output: saprot_control/data/saprot_embeddings.pt, saprot_control/data/saprot_index.csv
"""
import os, json, csv
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/20232202212/.cache/huggingface'
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, EsmForMaskedLM

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

BATCH_SIZE = 8
MAX_LENGTH = 1024
MODEL_NAME = 'westlake-repl/SaProt_650M_AF2'


def mean_pool(last_hidden_state, attention_mask):
    """Mean pool over non-padding tokens."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def main():
    print("=" * 70)
    print("Step 4: SaProt Inference")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Load sequences
    with open('saprot_control/data/saprot_sequences.json') as f:
        sequences = json.load(f)
    print(f"  Sequences to embed: {len(sequences)}")

    # Load model
    print(f"  Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmForMaskedLM.from_pretrained(MODEL_NAME)
    model = model.esm  # Use the encoder only
    model.eval()
    model.to(device)

    if device.type == 'cuda':
        model = model.half()
        print("  Using FP16")

    # Process in batches
    all_embeddings = []
    index_rows = []

    for batch_start in range(0, len(sequences), BATCH_SIZE):
        batch = sequences[batch_start:batch_start + BATCH_SIZE]
        saprot_seqs = []
        for item in batch:
            seq = item['saprot_sequence']
            # SaProt tokenizer expects space-separated tokens
            # Each token is 2 chars: AA(upper) + 3Di(lower)
            tokens = [seq[i:i+2] for i in range(0, len(seq), 2)]
            saprot_seqs.append(' '.join(tokens))

        inputs = tokenizer(
            saprot_seqs,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        ).to(device)

        with torch.no_grad():
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

            embeddings = mean_pool(
                outputs.last_hidden_state.float(),
                inputs['attention_mask']
            )

        all_embeddings.append(embeddings.cpu())

        for item in batch:
            index_rows.append({
                'seq_id': item['seq_id'],
                'label': item['label'],
                'analysis_label': item['analysis_label'],
                'aa_length': item['aa_length'],
            })

        if (batch_start // BATCH_SIZE) % 20 == 0:
            print(f"  Processed {batch_start + len(batch)}/{len(sequences)}")

    # Concatenate
    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    print(f"\n  Embeddings shape: {embeddings_tensor.shape}")

    # Validate
    assert embeddings_tensor.shape[0] == len(sequences), "Count mismatch!"
    assert not torch.isnan(embeddings_tensor).any(), "NaN detected!"
    assert not torch.isinf(embeddings_tensor).any(), "Inf detected!"
    print("  Validation: OK (no NaN/Inf)")

    # Save
    emb_path = Path('saprot_control/data/saprot_embeddings.pt')
    torch.save(embeddings_tensor, emb_path)
    print(f"  Saved: {emb_path}")

    idx_path = Path('saprot_control/data/saprot_index.csv')
    with open(idx_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['seq_id', 'label', 'analysis_label', 'aa_length'])
        w.writeheader()
        w.writerows(index_rows)
    print(f"  Saved: {idx_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
