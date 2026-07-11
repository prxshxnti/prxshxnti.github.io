---
layout: post
title: "LM Training Metrics"
date: 2026-06-15
excerpt: "Different training metrics wrt to model quality, throughput, compute, memory"
---

## Abstract

While training a model, we should track the variables/metrics that provide different information about the model we are training. The most commonly checked aspects are model quality, throughput, compute, and memory.

# Model Quality

Loss measures how far the model's predictions are from the true values. The following metrics test model quality. Generally, lower values of the metrics below indicate a better model.

1. **Training Loss**

Average training loss across all tokens in a batch. This measures how well the model is fitting the training dataset.

**Loss = Sum of token losses / Number of tokens**

2. **Validation Loss**

Average loss measured on unseen data (validation data). It detects generalization and overfitting. This loss is very important since it is evaluated on an unknown dataset.

3. **Perplexity**

It indicates the model's uncertainty when predicting the next token. It is an easier-to-interpret quality metric than loss.

Perplexity is the most interpretable version of cross-entropy loss. It measures how many possible next tokens the model is "choosing between" on average. Loss is what you optimize for, but perplexity is a more human-readable measure of uncertainty.

* Loss = 0, PPL = 1 (The model is certain and correct)
* Loss = 0.6, PPL = 2 (The model is choosing between 2 equally likely next tokens)
* Loss = 1.61, PPL = 5 (The model is choosing between 5 equally likely next tokens)

**Perplexity = e^Loss**

We can also track the best validation loss and save a checkpoint of that model.


# Throughput 


Throughput measures how quickly a model processes data during training. Monitoring throughput helps identify performance bottlenecks and evaluate hardware utilization.

1. Training Time per Epoch

An epoch represents one complete pass through the entire training dataset. During a single epoch, the model performs multiple parameter updates, where the number of updates equals the number of training steps in that epoch.

**Training Time per Epoch = Total Training Time / Number of Epochs**

A lower training time per epoch generally indicates faster training.

2. Step Time

Step time is the time required to complete a single training step, including: Forward pass, Backward pass, Gradient computation, Parameter update. The measurement is typically reported per batch. Lower step times indicate higher training efficiency.

Step Time = Time taken for one training iteration

3. Tokens per Second (Tokens/sec)

This metric measures how many tokens the model processes every second during training. It is one of the most common throughput metrics for language models. Higher values indicate better training throughput and more efficient hardware utilization.

**Tokens/sec = Total Tokens Processed / Training Time (seconds)**

4. Samples per Second (Samples/sec)

Samples per second measures the number of training sequences (samples) processed each second. This metric is useful for comparing training speed across different batch sizes and sequence lengths.

**Samples/sec = Total Number of Samples Processed / Training Time (seconds)**

# Compute 

1. Total compute / Flops / Floating point operations per second. 

The total number of floating-point calculations required for an entire training run. This is called as total compute required for the model. Estimating Flops for a transformer follows a common formula provide by chinchilla laws which is 

Flops = 6 * N * T 
where N is the number of model parameters, T is the total number of tokens process and 6 accounts for 2x operations in forward pass (one multiply and one add) and 4x operations in backward pass.  

We can also compute this analytically by breaking down the flops of every layer and accumulate it. We are measureing total training compute cost. 

Tflops = Total compute / 10^12

2. Tflops per sec / Achieved Tflops per sec / Effective Tflops per sec / Sustained Throughput 

Sustained Throughput  = Total compute / Training time (sec)
= 6 * N * T / training time.

This tells us how much compute the hardware actually delievered per second during training. Make sure the total compute is in teraflops unit.

3. GPU Uitlization 

GPU Uitlization = Sustained Throughput / GPU's maximum Tflops per sec.

We are calculating the Percentage of GPU compute being used. This tells how much of GPU is utilized and Reveals idle GPU time and bottlenecks

A low utilization often indicates bottlenecks such as:

Data loading delays
Communication overhead
Memory bandwidth limitations
Inefficient kernels

# Memory

1. Parameter memory 
It tells us the memory occupied by model weights and helps us determine the model size. 

2. Activation memory 
It tells us the memory used by intermediate layer outputs. 

3. Optimizer memory 
It tells us the emory used by optimizer states (Adam m, v, gradients). This is mostly the significant training memory overhead

4. Peak VRAM 
It provides us the maximum GPU memory used during training and let's us know where the training fits on available hardware. 


## Conclusion 

Quality metrics → Is the model learning?
Throughput metrics → How fast is training?
Compute metrics → How efficiently is hardware being used?
Memory metrics → What how space does our model needs?