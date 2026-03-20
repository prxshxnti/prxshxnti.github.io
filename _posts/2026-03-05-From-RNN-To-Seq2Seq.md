---
layout: post
title: "From RNN to Seq2Seq"
date: 2026-03-05
excerpt: "Covering all architectural advancements from RNN to Sequence-2-Sequence Model"
tags:
  - name: Deep Learning
    color: orange
---

Let us assume the core NLP problem we are facing right now is, we have sequences of data everywhere. We know how to process numeric data in deep learning, we know how to process image data in deep learning. What about textual data? 
Can a Fully connected network or a Convolutional Network understand sentences ? 

To understand more clearly, let's take a movie review, where we have to classify whether its negative feedback or positive feedback. We know that no movie reviews are expressed in numbers or image data. 
Movie reviews are textual content and such kind of information need to be processed in deep learning as well. There are lot of other tasks that has textual data, For example, Translation, Question and Answering, Text Summarization.

## Why Can't we process textual data using Fully connected Network ?

FC Network expects inputs in fixed size, we can't always give sentences in fixed number of words. That would kill the creativity in text. Positional order of the words is missed. Network can't capture temporal dependency, it doesn't have a time axis or a memory holding element.

## Recurrent Neural Networks 

This concept was introduced to make the neuron remember previous words. 

<figure>
  <img src="/images/rnn-2-sequence/rnn_equation.png" alt="RNN Equation">
  <figcaption>RNN Equation</figcaption>
</figure>

Example sentence: The movie Materialists is a great movie. 

Each word in the sentence is considered as a timestep. For first timestep the neuron processes 'The' word, In the second timestep the neuron processes 'movie' word. 
At the 6th time step, the neuron processes 'great' word. Neuron creates a memory vector (h_t) using the past memory (h_t-1) and current input (x_t) at each timestep. 

At the 6th time step, the past memory vector contains a compressed information of words from all 5 timesteps and it process 'great' word as x_T and generates the updated memory vector.

## Seq2Vec and Ses2Seq Architectures 

A sentence is a sequence and a word is a vector. 
1. For a task like sentiment classification of a movie review, the input is a sentence and output is a word (Negative/Positive). This architecture is called Sequence to Vector. 
It takes a review(Sequence) and gives out a sentiment(vector). 
2. For a task like translation from one language to another, the input is a sentence and the output is also a sentence. Example it takes a english sentence and gives out a german translated sentence , given that the task is german translation.
This architecture is called Sequence to Sequence.

For Seq2Vec architecture, refer to my blog on Sentiment classification. In the blog we will cover Seq2Seq architecture using a encoder-decoder system. 
The idea here is we have not used a complex NLP task like language translation, but a sequential task like Reverse a sequence to understand how the system's model temporal dependencies and understand long memory. 
In order to learn how these architectures(RNN -> LSTM -> GRU -> ATTENTION -> TRANSFORMER ), we have to use diagnostic experiments.
Trying to learn/train/develop a neural network on natural language with noisy, complex dataset is like introducing more complexity than just learning
In order to learn, it should involve controlled experiments, stimulated datasets created in the purpose to understand the mechanism

Now questions are :

Can this architecture store long-term information?

Can it align positions across sequences?

Can it generalize to longer inputs?

Natural language datasets are messy: multiple valid outputs, grammar variations, noise, ambiguous alignments. So improvements in architecture become hard to measure.
Algorithmic tasks are deterministic (clear correct answer), controlled difficulty (adjust sequence length [ train on <10 sequences, test on > 50 sequences. Rnn - 0%, lstm - 20%, attention - 50% and transformer-100%), interpretable ( We can clearly see the attention map)

## 1. Vanilla RNN

In our first architecture, we have used plain old RNN cell to build encoder and decoder. Lets talk about embeddings, we have words in sentence or a number in a list of numbers. Each unit is holding a meaning with respective of the task.
We cannot pass raw words / raw number to the model, we add a trainable embedding layer, that learns a vector capturing the semantic meaning of a token. 
Encoder takes in this sequence of embedding vectors( every word in a sentence is turned to a vector, sentence is a sequence) and generates 2 kinds of results.
A RNN cell gives outputs of neurons at each timesteps and final memory vector of last timestep. Remember output at a timestep is built using a memory vector at that timestep.

In the encoder we take a sequence and give out a vector. In the decoder we take a vector and give out a sequence. As per the code the encoder takes the embedding vectors and results the final hidden state. 
This final hiddens state is used in decoder to generate a sequence of outputs. Finally, the output layer assigns probability for each token in vocabulary based on previous computations.


```python
class ReverseTaskV0(nn.Module):
    """
    This Architecture involves the usage of a Encoder - Decoder system connected by a context vector
    The encoder/decoder architecture is build using a vanilla RNN
    """
    def __init__(self):
        super().__init__()
        self.enc_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.encoder = nn.RNN(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True)

        self.dec_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.decoder = nn.RNN(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True)
        self.output_layer = nn.Linear(in_features=HIDDEN_SIZE, out_features=VOCAB_SIZE)

    def forward(self, enc_inputs, dec_inputs):
        """
        :param enc_inputs: (B, S) -> 32, 10
        :param dec_inputs: (B, S) -> 32, 10
        :return: logits -> (B, S, Vocab_size) -> 32, 10, 20
        """
        enc_emb = self.enc_embeddings(enc_inputs) #(B, S, emb_dim) -> 32, 10, 32
        enc_out, enc_hidden = self.encoder(enc_emb) #(B, S, H), (L, B, H) -> (32, 10, 16), (1, 32, 16)

        dec_emb = self.dec_embeddings(dec_inputs)#(B, S, emb_dim) -> 32, 10, 32
        dec_out, dec_hidden = self.decoder(dec_emb, enc_hidden) #(B, S, H), (L, B, H) -> (32, 10, 16), (1, 32, 16)
        logits = self.output_layer(dec_out) #B, S, V -> 32, 10, 20
        return logits
```

Problem we face with this architecture is vanishing gradient problem. Long sentences cause the model to forget earlier words.
"The movie that I watched yesterday with my friend was amazing" . The word "movie" might influence "amazing", but RNN forgets it.

## 2. LSTM (Long Short-Term Memory)

Vanilla RNNs struggle with long sentences. They tend to forget earlier words due to the **vanishing gradient problem**. LSTM solves this by introducing **gates**.

### The Three Gates

- Forget Gate → decides what to erase  
- Input Gate → decides what to store  
- Output Gate → decides what to expose  

### Intuition

Think of LSTM as a smart notebook:
- It erases irrelevant information  
- It writes down important information  
- It reads selectively when needed  

This allows the model to remember important words even after many timesteps.

```python

class ReverseTaskV1(nn.Module):
    """
    This Architecture uses LSTM cell
    """
    def __init__(self):
        super().__init__()
        self.enc_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.encoder = nn.LSTM(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True)
        self.dec_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.decoder = nn.LSTM(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True)
        self.output_layer = nn.Linear(in_features=HIDDEN_SIZE, out_features=VOCAB_SIZE)

    def forward(self, enc_inputs, dec_inputs):
        enc_emb = self.enc_embeddings(enc_inputs)
        enc_out, (enc_hidden, enc_cell) = self.encoder(enc_emb)
        dec_emb = self.dec_embeddings(dec_inputs)
        dec_out, (dec_hidden, dec_cell) = self.decoder(dec_emb, (enc_hidden, enc_cell))
        logits = self.output_layer(dec_out)
        return logits

```
## 3. GRU (Gated Recurrent Unit)

LSTM solved the long-term memory problem in RNNs, but it introduced a more complex architecture with multiple gates.

GRU was proposed as a simpler alternative.

Instead of three gates (like in LSTM), GRU uses only two:
- Update gate → decides how much past information to keep  
- Reset gate → decides how much past information to forget  

### Intuition

If LSTM is a smart notebook with multiple control switches,  
GRU is a more streamlined version — fewer controls, but still effective.

It combines the memory and hidden state into a single representation, making it:
- simpler  
- faster to train  
- often comparable in performance to LSTM  

Because of this, GRUs are widely used when we want a balance between performance and efficiency.

```python

class ReverseTaskV2(nn.Module):
    """
    This Architecture uses GRU cell
    """
    def __init__(self):
        super().__init__()
        self.enc_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.encoder = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True)
        self.dec_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.decoder = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True)
        self.output_layer = nn.Linear(in_features=HIDDEN_SIZE, out_features=VOCAB_SIZE)

    def forward(self, enc_inputs, dec_inputs):
        enc_emb = self.enc_embeddings(enc_inputs)
        enc_out, enc_hidden = self.encoder(enc_emb)
        dec_emb = self.dec_embeddings(dec_inputs)
        dec_out, dec_hidden = self.decoder(dec_emb, enc_hidden)
        logits = self.output_layer(dec_out)
        return logits

```
## 4. Bidirectional GRU Encoder–Decoder

So far, we have seen that RNNs (and even LSTMs/GRUs) process a sentence **from left to right**. But language is not always understood in one direction.
For example: "I went to the bank to withdraw money" / "I sat near the river bank"
The meaning of the word *"bank"* depends not just on the past words, but also on the words that come after it.

### Bidirectional Encoder

To solve this, we use a **bidirectional GRU encoder**.
It reads the sentence in two directions:
- Forward (left → right)  
- Backward (right → left)

This gives us two hidden states: One capturing past context, One capturing future context. We combine these two to form a richer **context vector**.

### Connecting Encoder to Decoder

In this architecture:
- The encoder produces two final hidden states (forward + backward)
- We concatenate them to form a single context vector
- This context vector is used to initialize the decoder
Since the encoder is bidirectional, its output size doubles (2 × hidden size),so the decoder is designed to match this dimension.

### Intuition

Instead of understanding a sentence only from the past, the model now understands each word using **both past and future context**, leading to better sequence representations. Additionally, we would never have bidirectional encoders, because we always generate text from left to right 

```python
class ReverseTaskV3(nn.Module):
    """
    This architecture involves the use of bidirectional encoders using GRU cell
    """
    def __init__(self):
        super().__init__()
        self.enc_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.encoder = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True, bidirectional=True)
        self.dec_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.decoder = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=2*HIDDEN_SIZE, batch_first=True)
        self.output_layer = nn.Linear(2*HIDDEN_SIZE, VOCAB_SIZE)

    def forward(self, enc_inputs, dec_inputs):
        #(B, S) and (B, S)
        enc_emb = self.enc_embeddings(enc_inputs) #B, S, E
        enc_outputs, enc_hidden  = self.encoder(enc_emb) #(B, S, 2*H), (2, B, H)

        contex_vector = torch.cat([enc_hidden[0], enc_hidden[1]], dim=1) # (B, H ) + (B, H) along 1 = (B, 2*H)
        contex_vector = contex_vector.unsqueeze(0) #1, B, 2*H

        dec_emb = self.dec_embeddings(dec_inputs) #B, S, E
        dec_outputs, dec_hidden = self.decoder(dec_emb, contex_vector) #(B, S, 2*H), (1, B, 2*H)
        logits = self.output_layer(dec_outputs) #B, S, 2*H -> B, S, Vocabsize
        return logits
```
## 5. Bidirectional GRU with Bridge Layer

In the previous architecture, we used a bidirectional encoder and directly passed its concatenated hidden states (2 × hidden size) to the decoder. However, this creates a mismatch:
- Encoder produces a **larger representation (2H)**
- Decoder expects a **smaller hidden size (H)**

### Bridge Layer

To solve this, we introduce a **bridge layer**. The bridge is a simple linear transformation:
- It takes the concatenated encoder states (2H)
- Compresses them into a smaller vector (H)

### Why is this useful?
Instead of forcing the decoder to adapt to a larger hidden size, we **adapt the encoder output to match the decoder**.

### Intuition
Think of the bidirectional encoder as producing a **rich but large summary** of the sentence. The bridge layer acts like a **translator or compressor**, converting that rich representation into a form the decoder can efficiently use.

```python

class ReverseTaskV4(nn.Module):
    """
    This architecture involves the use of bidirectional encoders using GRU cell and bridge layer
    """
    def __init__(self):
        super().__init__()
        self.enc_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.encoder = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(2*HIDDEN_SIZE, HIDDEN_SIZE)
        self.dec_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.decoder = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True)
        self.output_layer = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)

    def forward(self, enc_inputs, dec_inputs):
        #(B, S) and (B, S)
        enc_emb = self.enc_embeddings(enc_inputs) #B, S, E
        enc_outputs, enc_hidden  = self.encoder(enc_emb) #(B, S, 2*H), (2, B, H)

        contex_vector = torch.cat([enc_hidden[0], enc_hidden[1]], dim=1) # (B, H ) + (B, H) along 1 = (B, 2*H)
        contex_vector = self.bridge(contex_vector) #B, H
        contex_vector = contex_vector.unsqueeze(0) #1, B, H

        dec_emb = self.dec_embeddings(dec_inputs) #B, S, E
        dec_outputs, dec_hidden = self.decoder(dec_emb, contex_vector) #(B, S, H), (1, B, H)
        logits = self.output_layer(dec_outputs) #B, S, H -> B, S, Vocab size
        return logits

```

## 6. Deep Bidirectional GRU (Stacked RNN Layers)

So far, we improved our model by:
- Adding bidirectionality → better context  
- Adding a bridge layer → better compatibility

Now, we take it one step further: **depth**.

### Why Deep RNNs?

A single RNN layer learns a representation of the sequence, but it may not capture all levels of abstraction.
By stacking multiple RNN layers:
- Lower layers capture simple patterns (local dependencies)  
- Higher layers capture more complex patterns (long-range relationships)

The encoder now has **3 stacked GRU layers**. Since it is bidirectional, we get: 3 layers × 2 directions = **6 hidden states**
Each layer produces its own hidden states,but not all are equally useful for decoding. We take only the **top-most layer's forward and backward states**:
These contain the most refined representation. We then Concatenate them → (2H) and Pass through the bridge → (H)

The decoder is also **3 layers deep**, so it expects an initial hidden state for each layer. To handle this: We duplicate the context vector across all decoder layers  

### Intuition

Think of this as a **hierarchical understanding of the sentence**, where the First layer captures basic word relationships, Second layer captures phrase-level understanding, Third layer captures full sentence meaning. 

```python

class ReverseTaskV5(nn.Module):
    """
    This architecture involves the use of bidirectional encoders using GRU cell and bridge layer and increased layers
    DEEP RNN
    """
    def __init__(self):
        super().__init__()
        self.enc_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.encoder = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True,
                              bidirectional=True, num_layers=3)
        self.bridge = nn.Linear(2*HIDDEN_SIZE, HIDDEN_SIZE)
        self.dec_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.decoder = nn.GRU(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True, num_layers=3)
        self.output_layer = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)

    def forward(self, enc_inputs, dec_inputs):
        #(B, S) and (B, S)
        enc_emb = self.enc_embeddings(enc_inputs) #B, S, E
        #for bidirectional GRU with one layer : 2 directions, bidirectional GRU with 3 layers - 6 directions ( num layers * num directions)
        enc_outputs, enc_hidden  = self.encoder(enc_emb) #(B, S, 2*H), (6, B, H)

        #Pick only the recent layers forward and backward states ( 4, 5 )
        contex_vector = torch.cat([enc_hidden[-2], enc_hidden[-1]], dim=1) # (B, H ) + (B, H) along 1 = (B, 2*H)
        contex_vector = self.bridge(contex_vector) #B, H
        contex_vector = contex_vector.unsqueeze(0) #1, B, H

        #Decoder architecture is 3 layers so expects contex vector of 3, B, H
        contex_vector = contex_vector.repeat(3, 1, 1) #duplicate a tensor along a dimension - here only repeat 3 times along first dimension

        dec_emb = self.dec_embeddings(dec_inputs) #B, S, E
        dec_outputs, dec_hidden = self.decoder(dec_emb, contex_vector) #(B, S, H), (1, B, H)
        logits = self.output_layer(dec_outputs) #B, S, H -> B, S, Vocab size
        return logits

```
That's it guys!! Thanks for reading. 


