---
layout: post
title: "Scaling Laws"
date: 2026-06-16
excerpt: "How do model size and training data size affect validation loss and perplexity for small language models?"
---

## Abstract 

Scaling Laws : 

Question : How do model size and training data size affect validation loss and perplexity for small language models?

# Initial Hypothesis 

H1: Parameter Scaling

Increasing parameters while keeping tokens fixed will reduce validation loss.

H2: Data Scaling

Increasing training tokens while keeping parameters fixed will reduce validation loss.

H3: Diminishing Returns

Gains from larger models and more data will eventually diminish.

H4: Compute Scaling

Validation loss approximately follows a power-law relationship with training compute.

# Experiment 

We are keeping a few things constant in our experiment namely Model Architecture, Number of layers, Optimizer, learning-rate, batch size, Tokenizer and Training procedure. Things that are varied are 

1. Parameters (1M, 3M, 10M, 30M)
2. Training Tokens (10M, 50M, 100M, 500M)

`We also have to find which combination of parameters and tokens is providing us the best result. `

This makes us perform 16 experiments and track the following metrics 
1. Parameter count
2. Training tokens
3. FLOPs estimate
4. Final validation loss
5. Final perplexity
6. Training time

Results: 
<figure>
  <img src="/images/scaling_laws/results_of_16_train_runs.png" alt="Results of 16 experiments">
  <figcaption>Results of 16 experiments</figcaption>
</figure>


# Analysis 

Plot A : Parameters vs Validation Loss
The tested parameter sizes are 1M, 3M, 10M, 30M

For each token budget:

1. 10M-token curve
2. 50M-token curve
3. 100M-token curve
4. 500M-token curve

<figure>
  <img src="/images/scaling_laws/model_size_vs_validation_loss.png" alt="Model size vs Validation Loss">
  <figcaption>Model size vs Validation Loss For a token budget</figcaption>
</figure>

Few things that we can notice are, 

At 10M tokens, 10M params is best, but 30M gets worse.
At 50M tokens, same thing.
At 100M tokens, 30M roughly ties 10M.
At 500M tokens, 30M finally becomes best.

This suggests the 30M model is becoming data-limited at small token budgets. A larger model has more capacity. If you don't give it enough data, it cannot fully utilize that capacity. Validation loss may plateau or even slightly worsen compared to a smaller model. Once you provide enough tokens, the larger model starts to pull ahead.

This is exactly the idea behind the compute-optimal scaling work from the Chinchilla scaling laws paper: larger models need proportionally more training data.

Our results suggest:

For 10M tokens, the optimal model is somewhere around 10M parameters. (These model size gives the lowest validation loss)
For 50M tokens, still around 10M parameters.
For 100M tokens, 10M–30M are similar.
For 500M tokens, 30M is finally justified.

That's already the beginning of a scaling-law analysis. The "turning point" moving rightward as token budget increases is exactly what you'd expect if larger models require more data to realize their advantage.

Question:
Does loss decrease as model size grows?

Answer: 
Not always. For a fixed token budget, validation loss decreases with model size only until a compute-optimal model size is reached. Beyond that point, larger models become data-limited and may not improve performance. As the token budget increases, this optimal model size shifts toward larger models, causing the loss curve to become increasingly monotonic. My results show this transition clearly: the 30M model underperforms at 10M and 50M tokens, matches the 10M model at 100M tokens, and becomes the best model at 500M tokens. This behavior is consistent with scaling-law predictions that larger models require proportionally more training data.


Plot B : Tokens vs Validation Loss
The tested token sizes are 10M, 50M, 100M, 500M

For each model size:

1. 1M curve
2. 3M curve
3. 10M curve
4. 30M curve

<figure>
  <img src="/images/scaling_laws/token_size_vs_validation_loss.png" alt="Token size vs Validation Loss">
  <figcaption>Token size vs Validation Loss For a token budget</figcaption>
</figure>

From our results:

Observation

For every model size:

- Increasing training tokens consistently reduces validation loss.
- No curve shows a reversal or plateau within the tested range.
- The largest improvement occurs when moving from 10M → 50M tokens.
- Additional gains continue at 100M and 500M tokens, though the improvement gradually diminishes.


Model-wise Analysis

1M Model - Loss decreases steadily:
1.75 → 1.45 → 1.37 → 1.25
This indicates the model continues to benefit from additional data throughout the tested range.

3M Model - Loss decreases monotonically:
1.66 → 1.38 → 1.31 → 1.19
The model effectively utilizes increasing amounts of training data.

10M Model - Loss decreases steadily:
1.63 → 1.37 → 1.28 → 1.16
No evidence of saturation is observed up to 500M tokens.

30M Model - Loss also decreases steadily:
1.67 → 1.40 → 1.28 → 1.15
Interestingly, the largest model gains the most from increased data, eventually outperforming all smaller models at 500M tokens.

Question:
Does more data always help?

Answer:
Yes. Validation loss decreases consistently as the number of training tokens increases for all tested model sizes. Larger datasets enable the models to learn more effectively and generalize better, resulting in lower validation loss. The improvement is particularly significant for larger models, which appear to benefit more from additional data. 


A nice additional insight from this plot is:

Data scaling is much more reliable than parameter scaling in my experiment.
Parameter scaling sometimes helped and sometimes hurt (depending on token budget), but increasing tokens always improved validation loss across all four model sizes.

Plot C : Compute vs Validation Loss

Estimate compute where ***Compute = parameters * tokens***
I have expanded our million notation(1M) into numeric values(1,000,000) since we need the true compute. Plotting compute in x-axis and validation loss at y-axis, I have received the following figure. 


<figure>
  <img src="/images/scaling_laws/compute_vs_validation_loss.png" alt="Compute vs Validation Loss">
  <figcaption>Compute vs Validation Loss For a token budget</figcaption>
</figure>

The observation we can make from this is 
1. For compute of 10^14, The loss of 1M model on a 100M token is 1.37 and loss of 10M model on a 10M token is 1.63. Surely the smaller model with more data of same compute achieves the least loss
2. For a compute of 3 * 10^14, the loss of 3M model on a 100M token data is 1.31 and loss of 30M model on a 10M data is 1.67. Like previous case small model with more data performs better than a large model with small data. 
3. For compute of 5 * 10^14, the loss of 1M model on 500M token is 1.25 and loss of 10M model on 50M tokens is 1.37. Even though the gap of loss is much smaller than previous example, Smaller model with more data outperforms a large model with its comparatively small data (50M data is small for 10M model)
4. For a compute of 15 * 10^14 or 1.5 * 10^15, The model with 3M Model size and 500M token size has a loss of 1.19 whereas the model with 30M model on 50M data has a loss of 1.40


For a single compute, since we have different loss, we are taking the best valid loss and plot the points again. 

<figure>
  <img src="/images/scaling_laws/compute_vs_validation_loss2.png" alt="Compute vs Best Validation Loss">
  <figcaption>Compute vs Best Validation Loss For a token budget</figcaption>
</figure>

Question:
Is there a smooth power-law trend?

Answer: 

The raw compute-versus-loss plot does not exhibit a smooth power-law trend because different allocations of compute produce different outcomes. However, the best-achievable-loss plot produces a better trend. The results also suggest that, for the compute budgets explored in this study, increasing training data is often more beneficial than increasing model size alone. However, the experiment contains only 16 runs, spanning four model sizes and four token budgets. A more larger experiment would definetly provide the necessary outcome.


# Compute Optimal 

Compute-optimal means the model size and training token count are balanced such that, for a fixed amount of compute (FLOPs)/Compute, you achieve the lowest possible validation loss.

Suppose you have enough compute for 100 GPU-hours.

You could train:

- A huge model on very little data.
- A tiny model on massive data.
- A medium-sized model on a medium amount of data.

The **compute-optimal point** is whichever combination gives the best validation loss for those 100 GPU-hours.

A useful intuition:

Too many parameters + too little data = data-limited.

Too much data + too few parameters = model-limited.

The sweet spot in between = compute-optimal

In our case : 

30M model on a 10M data - Data Limited (Because 30M performed better in 500M data)
500M data on a 1M model - Model Limited (Because 30M performed better in 500M data)
Sweet spot - 30M performed better in 500M data

Note : This is the closes to compute-optimal among the configurations I tested
It's possible an even better combination exists outside our grid.


# Conclusion: 

This study investigated the effect of model size and training data size on validation loss for small language models ranging from 1M to 30M parameters and trained on token budgets ranging from 10M to 500M tokens.

The results show that both model scaling and data scaling improve performance, but their effects are not equally reliable. Increasing model size reduces validation loss only up to a certain point. For small token budgets (10M and 50M tokens), larger models eventually become data-limited, causing performance gains to diminish or even reverse. However, as the token budget increases, the optimal model size shifts toward larger models. At 500M tokens, the 30M-parameter model achieves the lowest validation loss among all tested configurations.

In contrast, increasing the amount of training data consistently improves performance across all tested model sizes. No model exhibited saturation within the explored token range. This suggests that, for the scales studied here, additional training data is a more reliable source of performance improvement than increasing parameter count alone.

The compute analysis further demonstrates that compute allocation matters. Models trained with the same approximate compute budget can achieve substantially different validation losses depending on how compute is distributed between parameters and training tokens. In several cases, smaller models trained on larger datasets outperformed larger models trained on smaller datasets despite using the same total compute. 

Overall, the experiment supports all four initial hypotheses. Parameter scaling improves performance when sufficient data is available, data scaling consistently improves performance, diminishing returns emerge as models become data-limited, and the compute frontier exhibits qualitative scaling-law behavior. These results demonstrate that even small language models exhibit many of the same scaling trends observed in modern large language models.


# Notes for me - my pain points 

Finding the appropriate dataset for my experiments.
In HF, the datasets were present in parquet files and it was hard to convert them to csv. Using pandas, each row was written into text file. 
Some files were too big that didn't even load and kernel crashed, batch reading didn't work.  Solved with HF datasets direct download
Some parquet files has been corrupted so its not reading. Solved with HF datasets direct download
