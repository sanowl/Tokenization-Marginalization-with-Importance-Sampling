




# Tokenization Marginalization with Importance Sampling

This project provides a Python implementation of an importance sampling algorithm to estimate the marginalized probability over possible tokenizations using the GPT-2 language model.

## Overview

Autoregressive language models typically compute the probability of a string by first transforming it into a sequence of tokens. However, there can be exponentially many possible tokenizations for any given string. To compute the true probability, one should marginalize over all tokenizations. This project implements an importance sampling-based algorithm to estimate these marginalized probabilities.

## Features

- Uses the GPT-2 language model from Hugging Face's `transformers` library.
- Implements importance sampling to handle multiple tokenizations.
- Computes log probabilities of sequences and estimates marginalized probabilities.

## Installation

1. Clone the repository:
    ```sh
    https://github.com/sanowl/Tokenization-Marginalization-with-Importance-Sampling.git
    cd tokenization-marginalization
    ```

2. Install the required packages:
    ```sh
    pip install transformers torch numpy
    ```

## Usage

1. Run the script with a sample text:
    ```sh
    python main.py
    ```

2. Customize the text and number of samples by modifying the `main.py` script.

## Example

The following code snippet demonstrates how to use the importance sampling function:

```python
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load the model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

def tokenize(text, tokenizer):
    """Tokenize the input text."""
    return tokenizer.tokenize(text)

def detokenize(tokens, tokenizer):
    """Detokenize the tokens back into a string."""
    return tokenizer.convert_tokens_to_string(tokens)

def log_probability(tokens, model, tokenizer):
    """Compute the log probability of a sequence of tokens."""
    inputs = tokenizer(tokens, return_tensors='pt', is_split_into_words=True)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    log_probs = outputs.logits.log_softmax(dim=-1)
    token_ids = inputs["input_ids"].squeeze()
    token_log_probs = log_probs[0, torch.arange(len(token_ids)), token_ids]
    return token_log_probs.sum().item()

def importance_sampling(text, tokenizer, model, num_samples=100):
    """Estimate the marginalized probability using importance sampling."""
    original_tokens = tokenize(text, tokenizer)
    original_log_prob = log_probability(original_tokens, model, tokenizer)

    tokenizations = [tokenize(text, tokenizer) for _ in range(num_samples)]
    log_probs = np.array([log_probability(t, model, tokenizer) for t in tokenizations])
    
    weights = np.exp(log_probs - original_log_prob)
    normalized_weights = weights / np.sum(weights)
    
    estimated_log_prob = np.sum(normalized_weights * log_probs)
    
    return estimated_log_prob

def main():
    text = "This is an example sentence for testing."
    num_samples = 100

    estimated_log_prob = importance_sampling(text, tokenizer, model, num_samples=num_samples)
    print(f"Estimated log probability: {estimated_log_prob}")

if __name__ == "__main__":
    main()
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hugging Face for the `transformers` library.
- The authors of the paper "Should you marginalize over possible tokenizations?" for the theoretical foundation.
