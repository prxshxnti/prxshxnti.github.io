---
layout: post
title: "Character LM Inference"
date: 2026-06-07
excerpt: "Exposing char LM as an API with different decoding strategies"
---

## Abstract 

In the entry, I will be discussing about how I have exposed my Language Model as an API and how inference is performed on the model. Additionally, I will cover the decoding strategies that I have experimented with.

## Inference 

During training, I stored the necessary model artifacts, including the vocabulary, metadata (number of layers, heads, and sequence length), and the model weights corresponding to the best validation loss. Using this information, I am able to reconstruct the exact model that achieved the best performance.

I exposed the inference class through a FastAPI application running on a Uvicorn server. Pydantic models are used to validate incoming requests and return structured responses. When the FastAPI application starts, the model weights and all required artifacts are loaded, the model is reconstructed, and it is set to evaluation mode to serve inference requests.

When a request is received at the `/generate` endpoint, the input is validated to ensure it follows the expected request structure. To prevent unlimited token generation, the requested output length is constrained to a predefined limit. For every token in the prompt, I verify that it exists in the vocabulary, as there is no unknown token in the current implementation. The context is formed by selecting the last few tokens up to the sequence length. This context is then passed to the model to predict the next token. The model outputs logits, which represent token probabilities, and various decoding strategies can be used to select the next token.

##  Infrastructure and Exposing as an API

My setup consists of a remote machine running Docker, with all required dependencies installed within the container. I chose to expose the API from the container rather than the host machine because the inference application depends on the containerized environment and its dependencies. Using host-to-container port mappings, the API is exposed through a Uvicorn server.

## Decoding Strategies

During next-token prediction, model gives the probabilities of every token in the vocabulary. This probability defines how more likely a token is to appear next. Different decoding stragies play differently with this probability distribution. Next token sampling is performing differently as mentioned below, 

# 1. Greedy Decoding 

Our vocabulary size is 65, and model gives probabilities for all 65 tokens and only one token out of it has the maximum probability and greedy decoding chooses that. The process of choosing the highest-probability token at each step is called greedy decoding. 

For example : a = 0.3, e = 0.8, i = 0.1, o = 0.4, u = 0.6


Out of the 5 tokens, token `e` has the maximum prob and hence its choosen as the next token

```python
next_token = logits.argmax().item()
```

This approach is fastest, deterministic and easier to implement. 

# 2. Temperature sampling

This is a hyperparameter that can be provided during inference to shape the distribution of logits and probabilities. Before converting the logits to probabilities, we can divide them by a factor called temperature.

When this factor is equal to 1, we get the original probabilities with no modification to the probability distribution of the tokens.

Using a value less than 1 makes the distribution sharper, and the output becomes more deterministic since the gaps between token probabilities become wider.

Using a value greater than 1 flattens the distribution, giving more tokens a higher chance of being chosen, making the output more stochastic.

```python 

logits_with_temp = logits / temperature 
probs_with_temp = torch.softmax(logits_with_temp, dim=-1)
next_token = torch.multinomial(probs_with_temp, num_samples = 1).item()

```

For example : 

Logits: tensor([9., 7., 6.])

Probabilities with Temp = 1:
tensor([0.8438, 0.1142, 0.0420])

Temp 0.2:
tensor([1.0000, 0.0000, 0.0000])

Temp 0.6:
tensor([0.9593, 0.0342, 0.0065])

Temp 4:
tensor([0.4810, 0.2918, 0.2272])

Temp 8:
tensor([0.4055, 0.3158, 0.2787])
 
The higher the temperature, the flatter the probability distribution becomes. The lower the temperature, the more concentrated the probability mass becomes around the most likely token.

# 3. Top - K sampling 

Top - K is a hyperparameter that chooses the candidate tokens for sampling, ex : top k set to 20 will consider the probs of top 20 tokens and this discards the low-probability tokens. 

```python

values, indices = torch.topk(logits, k)

filtered = torch.full_like(logits, float('-inf'))
filtered[indices] = values 

probs = torch.softmax(filtered, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)

```
By assigning `-inf` to all tokens outside the top `k`, their probabilities become zero after applying softmax. The next token is then sampled only from the remaining candidates.

This method is often used together with **temperature**. Temperature controls the shape of the probability distribution, while top-k restricts the set of candidate tokens that can be sampled.

One advantage of top-k sampling is that it prevents the model from selecting extremely unlikely tokens, which can improve the quality and coherence of generated text.

A limitation of top-k sampling is that the value of `k` is fixed and must be chosen empirically. In some contexts, there may be only a few reasonable candidate tokens, while in others there may be hundreds. A fixed `k` cannot adapt to these situations, which is one reason why **top-p (nucleus) sampling** is often preferred in modern language models.

# 4. Top - P sampling 

Unlike Top-k sampling, which always keeps a fixed number of candidate tokens, Top-p sampling dynamically adjusts the number of candidate tokens based on the model's confidence. If the model is very confident, only a few tokens may be kept. If the model is uncertain, many tokens may be retained. This adaptive behavior often produces more natural and coherent text generation, which is why Top-p sampling is widely used in modern LLMs.

```python 

sorted_logits , sorted_indices = torch.sort(logits, descending=True) # Sort logits from highest to lowest
sorted_probs = torch.softmax(sorted_logits, dim=-1)  # Convert sorted logits to probabilities
cum_probs = torch.cumsum(sorted_probs, dim=-1) # e.g. [0.5, 0.2, 0.1, 0.05, ... becomes [0.5, 0.7, 0.8, 0.85, ...]

remove = cum_probs > top_p # Example: top_p = 0.8, cum_probs = [0.5, 0.7, 0.8, 0.9, 1.0]
#remove = [False, False, False, True, True] We want to keep the token that first reaches/exceeds top_p,
# so we shift the mask one position to the right.
remove[1:] = remove[:-1].clone()
remove[0] = False # Always keep the highest-probability token.

# The tokens marked False form the nucleus.
# Their cumulative probability is at least top_p.
sorted_logits[remove] = float('-inf') # Set removed tokens to -inf so their softmax probability becomes 0.
filtered = torch.softmax(sorted_logits, dim=-1) 

sampled = torch.multinomial(filtered, num_samples=1) #Sample one token from the nucleus distribution
next_token = sorted_indices[sampled].item()

```

# 5. Beam search 

Beam Search is a decoding strategy that keeps track of the top k most likely sequences (not tokens) at each generation step instead of considering only the single most likely token.

Greedy decoding keeps only 1 sequence. Beam search keeps k sequences. Beam is a collection of sequence and score. At each step, every beam's sequence is expanded with possible next tokens. The top k sequences with the highest cumulative scores are retained.

Beam search maintains multiple candidate sequences during decoding and selects the sequence with the highest overall probability, reducing the risk of getting stuck with a poor early token choice as in greedy decoding.

```python

beams = [(sequence, score)]
beam_size = 5

for step in range(max_new_tokens):

    candidates = []

    for seq, score in beams:

        logits = model(seq)

        probs = softmax(logits)

        top_probs, top_ids = torch.topk(probs, beam_size)

        for p, idx in zip(top_probs, top_ids):
            candidates.append(
                (seq + [idx], score + torch.log(p))
            )

    beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
```


