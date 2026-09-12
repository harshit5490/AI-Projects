# Transformers — Complete Notes

## 1. Sequence-to-Sequence Problem

### What is a Sequence?

A sequence is ordered data where the order of elements matters.

Examples:
- Natural language sentences
- Time-series data
- Audio
- Machine translation input/output

Example:

```text
I love machine learning
```

The order of the tokens carries meaning.

### What is Sequence-to-Sequence (Seq2Seq)?

A Seq2Seq system takes one sequence as input and produces another sequence as output.

```text
Input Sequence
      ↓
   Encoder
      ↓
Representation
      ↓
   Decoder
      ↓
Output Sequence
```

Examples:
- Machine translation
- Text summarization
- Question answering
- Conversational systems

### Traditional Encoder-Decoder Architecture

The encoder reads the input sequence and creates a representation. The decoder uses that representation to generate the output sequence.

A major limitation of early encoder-decoder systems was the need to compress the input into a representation, especially for long sequences.

This motivated the development of attention.

---

# 2. Attention Intuition

Attention allows a model to determine which parts of a sequence are more relevant when processing a particular token.

Instead of treating every token equally, the model can assign different weights to different tokens.

Example:

```text
The cat sat on the mat.
```

When processing `sat`, the model may learn stronger relationships with relevant tokens such as `cat`.

Conceptually:

```text
Input tokens
     ↓
Calculate relevance
     ↓
Attention weights
     ↓
Weighted information
     ↓
New representation
```

### Attention Weights

Attention weights indicate how much importance one token gives to other tokens.

They are commonly normalized using softmax:

```text
Raw scores
    ↓
 Softmax
    ↓
Attention weights
```

The weights form a probability-like distribution.

### Attention vs Self-Attention

**Attention** is the general mechanism.

**Self-attention** is a specific form where Query, Key, and Value are derived from the same sequence.

Cross-attention is another important form where, in the original encoder-decoder Transformer, the Query comes from the decoder while Key and Value come from the encoder.

---

# 3. Query, Key, Value (Q/K/V)

Q/K/V is the core mechanism used to calculate attention.

### Intuition

- **Query (Q):** What information am I looking for?
- **Key (K):** What information do I contain / how can I be matched?
- **Value (V):** What information should I provide?

### Learned Projections

For an input representation `X`:

```text
Q = XWQ
K = XWK
V = XWV
```

where `WQ`, `WK`, and `WV` are learned projection matrices.

### Scaled Dot-Product Attention

The main formula is:

```text
Attention(Q, K, V)
=
softmax(QKᵀ / √dk)V
```

Steps:

```text
Q and K
   ↓
Dot product
   ↓
QKᵀ
   ↓
Scale by √dk
   ↓
Softmax
   ↓
Attention weights
   ↓
Multiply by V
   ↓
Attention output
```

### Why Divide by √dk?

If the Key/Query dimension becomes large, dot products can become large.

Large scores can make softmax very peaked, which can lead to poor gradients.

Dividing by:

```text
√dk
```

helps keep the values at a more manageable scale.

### Educational NumPy Implementation

```python
import numpy as np


def softmax(x):
    exp_x = np.exp(
        x - np.max(x, axis=-1, keepdims=True)
    )

    return exp_x / np.sum(
        exp_x,
        axis=-1,
        keepdims=True
    )


Q = np.array([
    [1.0, 0.0],
    [0.0, 1.0]
])

K = np.array([
    [1.0, 0.0],
    [0.0, 1.0]
])

V = np.array([
    [10.0, 0.0],
    [0.0, 20.0]
])


scores = Q @ K.T

dk = K.shape[-1]

scaled_scores = scores / np.sqrt(dk)

attention_weights = softmax(scaled_scores)

output = attention_weights @ V
```

### Important

This is an educational implementation.

A real Transformer learns separate Q/K/V projection matrices:

```text
Q = XWQ
K = XWK
V = XWV
```

---

# 4. Self-Attention From Start to Finish

Self-attention allows every token to interact with other tokens in the same sequence.

Example:

```text
I love AI
```

Represent the tokens using a toy matrix:

```python
X = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])
```

For demonstration, we can temporarily use:

```text
Q = X
K = X
V = X
```

Then:

```text
QKᵀ
=
[[1, 0, 1],
 [0, 1, 1],
 [1, 1, 2]]
```

After scaling by `√2`:

```text
[[0.707, 0, 0.707],
 [0, 0.707, 0.707],
 [0.707, 0.707, 1.414]]
```

Approximate softmax attention weights:

```text
[[0.401, 0.198, 0.401],
 [0.198, 0.401, 0.401],
 [0.248, 0.248, 0.503]]
```

Then:

```text
Output = AttentionWeights × V
```

### Educational Code

```python
import numpy as np


def softmax(x):
    exp_x = np.exp(
        x - np.max(x, axis=-1, keepdims=True)
    )

    return exp_x / np.sum(
        exp_x,
        axis=-1,
        keepdims=True
    )


X = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])


Q = X
K = X
V = X


scores = Q @ K.T

dk = K.shape[-1]

scaled_scores = scores / np.sqrt(dk)

attention_weights = softmax(scaled_scores)

output = attention_weights @ V
```

### Important Distinction

The above uses:

```text
Q = K = V = X
```

only to make the mathematics easy to understand.

A real Transformer uses:

```text
Q = XWQ
K = XWK
V = XWV
```

where the projection matrices are learned.

---

# 5. Multi-Head Attention

One attention operation may not capture all useful relationships in a representation.

Multi-head attention allows multiple attention operations to work in parallel.

Each head has separate projections:

```text
Qi = XWQi
Ki = XWKi
Vi = XWVi
```

Each head performs scaled dot-product attention.

```text
Head 1 ──┐
Head 2 ──┤
Head 3 ──┼──→ Concatenate → WO → Output
...     ──┤
Head h ──┘
```

Formula:

```text
MultiHead(Q,K,V)
=
Concat(head1,...,headh)WO
```

where:

```text
headi = Attention(Qi, Ki, Vi)
```

### Head Dimension

Typically:

```text
d_head = d_model / number_of_heads
```

Example:

```text
d_model = 512
heads = 8

d_head = 512 / 8
       = 64
```

### Important

Do not assume that a particular head always means something like "grammar head" or "subject head." Attention patterns are learned and can be complex.

---

# 6. Positional Information

Attention itself does not inherently encode the order of tokens.

Consider:

```text
Dog bites man
```

and:

```text
Man bites dog
```

The same words are present, but the order changes the meaning.

Therefore Transformers need positional information.

### Original Transformer Sinusoidal Positional Encoding

The original Transformer used:

```text
PE(pos, 2i)
=
sin(pos / 10000^(2i/d_model))
```

and:

```text
PE(pos, 2i+1)
=
cos(pos / 10000^(2i/d_model))
```

The positional encoding is added to the token embedding:

```text
Token Embedding
      +
Position Encoding
      ↓
Input Representation
```

The dimensions must match.

### Educational Example

```python
import numpy as np


token_embedding = np.array([
    [0.2, 0.7, 0.1, 0.4],
    [0.5, 0.1, 0.8, 0.3],
    [0.9, 0.2, 0.4, 0.6]
])


position_encoding = np.array([
    [0.1, 0.3, 0.2, 0.5],
    [0.2, 0.4, 0.1, 0.3],
    [0.3, 0.1, 0.4, 0.2]
])


input_representation = (
    token_embedding + position_encoding
)
```

### Other Positional Approaches

Different models use different approaches.

Examples:

- Original Transformer: sinusoidal positional encoding
- BERT: learned positional embeddings
- Llama-style models: RoPE (Rotary Position Embedding)

The exact positional mechanism depends on the architecture/model generation.

---

# 7. Encoder vs Decoder

The original Transformer uses an encoder-decoder architecture.

### Encoder

The encoder processes the input sequence and creates contextual representations.

```text
Input
 ↓
Encoder
 ↓
Contextual representation
```

### Decoder

The decoder generates the output sequence.

```text
Previous output tokens
        ↓
     Decoder
        ↓
Next token
```

### Encoder Block

Conceptually:

```text
Input
 ↓
Multi-Head Self-Attention
 ↓
Residual + LayerNorm
 ↓
Feed-Forward Network
 ↓
Residual + LayerNorm
 ↓
Output
```

### Decoder Block

Conceptually:

```text
Input
 ↓
Masked Self-Attention
 ↓
Add + Norm
 ↓
Cross-Attention ← Encoder Output
 ↓
Add + Norm
 ↓
Feed-Forward Network
 ↓
Add + Norm
 ↓
Output
```

### Masked Self-Attention

A decoder should not see future tokens during autoregressive generation.

For:

```text
I love AI
```

the causal visibility pattern is:

```text
        I   love   AI
I       ✓    ✗     ✗
love    ✓    ✓     ✗
AI      ✓    ✓     ✓
```

This prevents future-token information from leaking into the current prediction.

### Cross-Attention

In the original encoder-decoder Transformer:

```text
Query → Decoder
Key   → Encoder
Value → Encoder
```

Cross-attention allows the decoder to use information from the encoder.

---

# 8. Transformer Architecture Families

### Encoder-only

Example:

```text
BERT
```

Characteristics:

- Encoder architecture
- Bidirectional attention
- Strong for understanding/representation tasks
- Uses Masked Language Modeling during pretraining

### Decoder-only

Examples:

```text
GPT-style models
Llama-style models
```

Characteristics:

- Decoder architecture
- Causal self-attention
- Autoregressive next-token prediction
- Strong for generation

### Encoder-decoder

Examples include the original Transformer and T5-style architectures.

Characteristics:

- Encoder processes input
- Decoder generates output
- Decoder can use encoder information through cross-attention
- Useful for sequence-to-sequence tasks

---

# 9. Transformer Block

A Transformer block is a reusable processing unit.

Multiple blocks are stacked to create a Transformer model.

## Encoder Block

```text
Input
  ↓
Self-Attention
  ↓
Residual + Normalization
  ↓
FFN
  ↓
Residual + Normalization
  ↓
Output
```

## Decoder Block

```text
Input
  ↓
Masked Self-Attention
  ↓
Residual + Normalization
  ↓
Cross-Attention
  ↓
Residual + Normalization
  ↓
FFN
  ↓
Residual + Normalization
  ↓
Output
```

### Residual Connection

A residual connection adds the original input back:

```text
Output = X + Attention(X)
```

It provides a direct information/gradient path and helps deep networks train effectively.

### LayerNorm

Layer Normalization helps stabilize representations and training.

LayerNorm and BatchNorm are not the same.

Modern Transformer architectures can use different normalization arrangements, including **Pre-Norm** and **Post-Norm**.

### Feed-Forward Network (FFN)

A typical FFN is:

```text
FFN(x)
=
activation(xW1 + b1)W2 + b2
```

Conceptually:

```text
Input
 ↓
Linear
 ↓
Activation
 ↓
Linear
 ↓
Output
```

Common activations include GELU and, in modern architectures, gated variants such as SwiGLU.

The FFN operates independently at each token position using shared parameters.

The attention mechanism handles communication between token positions.

### Core Roles

```text
Attention
→ token-to-token information exchange

FFN
→ per-token nonlinear transformation

Residual
→ preserves information/direct path

Normalization
→ stabilizes representations/training
```

---

# 10. BERT + Masked Language Modeling

BERT stands for:

**Bidirectional Encoder Representations from Transformers**

BERT is an **encoder-only Transformer**.

### Bidirectional

When creating a representation for a token, BERT can use context from both the left and right sides.

Example:

```text
The cat is [MASK] on the mat.
```

The model uses surrounding context to predict the masked token.

### Masked Language Modeling (MLM)

During pretraining, selected tokens are hidden and the model learns to predict them.

Example:

```text
The cat is [MASK] on the mat.
```

Possible prediction:

```text
sitting
```

### Original BERT Masking Recipe

For selected tokens, the commonly described original recipe was approximately:

```text
80% → replace with [MASK]
10% → replace with a random token
10% → keep unchanged
```

### BERT Architecture

```text
Input tokens
     ↓
Token + Position representations
     ↓
Transformer Encoder blocks
     ↓
Contextual representations
```

BERT is useful for tasks such as:

- Text classification
- Sentiment analysis
- Named Entity Recognition
- Extractive Question Answering
- Sentence similarity

### Extractive QA

BERT-style extractive QA can predict the start and end positions of an answer span in a provided context.

It does not have to generate the answer token-by-token.

### BERT vs GPT

| Feature | BERT | GPT-style |
|---|---|---|
| Architecture | Encoder-only | Decoder-only |
| Attention | Bidirectional | Causal |
| Pretraining objective | MLM | Next-token prediction |
| Main strength | Understanding/representations | Generation |
| Future tokens visible? | Yes during encoder representation | No during causal generation |

---

# 11. GPT-Style Decoder-Only Architecture

GPT stands for:

**Generative Pre-trained Transformer**

GPT-style models are decoder-only Transformers.

### Autoregressive Prediction

The model predicts the next token based on previous tokens:

```text
P(next token | previous tokens)
```

For a sequence:

```text
x1, x2, ..., xn
```

the autoregressive factorization is:

```text
P(x1,...,xn)
=
∏ P(xt | x1,...,x(t-1))
```

### Causal Mask

The model must not use future tokens.

```text
        I   love   AI
I       ✓    ✗     ✗
love    ✓    ✓     ✗
AI      ✓    ✓     ✓
```

Future attention scores can conceptually be replaced by:

```text
-inf
```

before softmax.

After softmax, those positions receive probability approximately zero.

### GPT-Style Decoder Block

A simplified decoder-only block:

```text
Input
 ↓
Causal Self-Attention
 ↓
Residual + Normalization
 ↓
FFN
 ↓
Residual + Normalization
 ↓
Output
```

There is no encoder and therefore no encoder-decoder cross-attention in a standard decoder-only GPT-style architecture.

### Generation

```text
Prompt
 ↓
Predict next token
 ↓
Append token
 ↓
Predict next token
 ↓
Append token
 ↓
Repeat
```

Important distinction:

- During training, many target positions can be processed in parallel using causal masking.
- During autoregressive inference, generation proceeds token by token.

### Educational Causal Mask Code

```python
import numpy as np


scores = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
])


mask = np.triu(
    np.ones(scores.shape),
    k=1
)


masked_scores = np.where(
    mask == 1,
    -np.inf,
    scores
)


print(masked_scores)
```

`np.triu(..., k=1)` identifies the positions above the main diagonal.

Those future positions are replaced with `-inf`.

---

# 12. Llama Overview

Llama is a family of large language models from Meta.

Llama-style models are **decoder-only Transformers**.

The core idea remains:

```text
Causal Self-Attention
+
Autoregressive Next-Token Prediction
```

Llama-style architectures refine the Transformer design with modern components.

### Important Components

#### RoPE

**Rotary Position Embedding**

RoPE incorporates positional information through rotations applied to representations, especially Query and Key representations.

It helps encode positional/relative relationships.

#### RMSNorm

**Root Mean Square Normalization**

RMSNorm is a normalization method based on the root mean square of activations and does not perform the same mean-centering operation as LayerNorm.

#### SwiGLU

A gated feed-forward architecture using the SiLU/Swish activation.

It provides a more expressive FFN structure.

### Simplified Llama-Style Block

```text
Input
 ↓
RMSNorm
 ↓
Causal Self-Attention + RoPE
 ↓
Residual Add
 ↓
RMSNorm
 ↓
SwiGLU
 ↓
Residual Add
 ↓
Output
```

Exact architectural details can vary across Llama generations.

### Comparison

| Architecture | Main characteristics |
|---|---|
| Original Transformer | Encoder + Decoder, sinusoidal positions, LayerNorm, FFN |
| BERT | Encoder-only, bidirectional, MLM |
| GPT-style | Decoder-only, causal, next-token prediction |
| Llama-style | Decoder-only, causal, RoPE, RMSNorm, SwiGLU |

As an AI Engineer, the goal is to understand these architectural concepts rather than implement every large model from scratch.

---

# 13. Tokenization + Subword Tokens

Transformers do not directly process raw text.

The pipeline is:

```text
Raw Text
   ↓
Tokenization
   ↓
Tokens
   ↓
Token IDs
   ↓
Embeddings
   ↓
Transformer
```

### Token

A token can be:

- Complete word
- Subword
- Punctuation
- Special token
- Other tokenizer-defined unit

A token is not necessarily a word.

### Why Not Word-Level Tokenization?

Problems include:

- Huge vocabulary
- Unknown words
- Poor handling of rare/new words

### Why Not Character-Level?

Character-level tokenization can create extremely long sequences.

### Subword Tokenization

Subword tokenization provides a balance between vocabulary size and handling unknown/rare words.

Examples:

- BPE
- WordPiece
- SentencePiece-based approaches

### BPE

Byte Pair Encoding-style tokenization repeatedly merges frequent units.

Conceptually:

```text
Small units
 ↓
Find frequent pairs
 ↓
Merge
 ↓
Repeat
 ↓
Subword vocabulary
```

### WordPiece

WordPiece is associated strongly with BERT.

Continuation pieces may use a convention such as:

```text
##ing
```

### SentencePiece

SentencePiece performs subword tokenization without depending on traditional whitespace splitting.

The `▁` symbol is commonly used to represent a word boundary in SentencePiece token representations.

### Special Tokens

Depending on the model:

```text
[CLS]
[SEP]
[MASK]
[PAD]
<BOS>
<EOS>
```

can be used.

### Token IDs

Token IDs are vocabulary indexes.

They are not semantic numerical values.

Example:

```text
"hello"
   ↓
token ID 7592
```

The specific ID depends on the tokenizer vocabulary.

### Padding

Padding makes sequences in a batch the same length.

```text
[CLS] I love AI [SEP] [PAD] [PAD]
```

### Attention Mask

Often:

```text
1 → real token
0 → padding
```

### Padding Mask vs Causal Mask

These should not be confused.

**Padding attention mask:**

```text
Ignore padding tokens.
```

**Causal mask:**

```text
Prevent access to future tokens.
```

### Tokenizer vs Model

Tokenizer:

```text
Text → token IDs
```

Model:

```text
Token IDs → representations/predictions
```

---

# 14. Hugging Face Transformers

Hugging Face `transformers` provides tools for:

- Pretrained Transformer models
- Tokenizers
- Inference
- Pipelines
- Training/fine-tuning
- Text generation
- Many task-specific model classes

### Installation

```bash
pip install transformers torch
```

Verify:

```bash
python -c "import transformers; print(transformers.__version__)"
```

### Loading a Tokenizer

```python
from transformers import AutoTokenizer


model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)
```

`AutoTokenizer` automatically selects the appropriate tokenizer implementation for the checkpoint.

### Tokenize Text

```python
text = "I love machine learning"

tokens = tokenizer.tokenize(text)

print(tokens)
```

### Encode Text

```python
inputs = tokenizer(
    text,
    return_tensors="pt"
)
```

This commonly returns fields such as:

```text
input_ids
attention_mask
```

Some models can also return:

```text
token_type_ids
```

### `input_ids`

Integer IDs representing tokens.

### `attention_mask`

Indicates which positions should be treated as valid/attended to, especially for padding.

### `token_type_ids`

Can identify segments in models such as BERT, but not every Transformer model uses them.

### Load a Base Model

```python
from transformers import AutoModel


model = AutoModel.from_pretrained(
    model_name
)
```

### Run the Model

```python
import torch


with torch.no_grad():
    outputs = model(**inputs)
```

`**inputs` unpacks the dictionary into keyword arguments.

### Hidden State

For many base Transformer models:

```python
outputs.last_hidden_state
```

has a shape like:

```text
[batch_size, sequence_length, hidden_size]
```

For BERT base:

```text
hidden_size = 768
```

Example:

```text
[1, 6, 768]
```

means:

```text
1 sequence
6 tokens
768 features per token
```

### Task-Specific Classes

Examples:

```text
AutoModel
AutoModelForSequenceClassification
AutoModelForQuestionAnswering
AutoModelForCausalLM
```

Use the task-specific class when you need task-specific outputs.

---

# 15. Loading Pretrained Models

A pretrained model has already learned weights from large-scale training.

### `from_pretrained()`

Conceptually:

```text
Identify checkpoint
      ↓
Download/cache files
      ↓
Read configuration
      ↓
Construct architecture
      ↓
Load pretrained weights
```

Example:

```python
from transformers import AutoModel


model = AutoModel.from_pretrained(
    "bert-base-uncased"
)
```

### Model Configuration

A model configuration can contain information such as:

```text
hidden_size
num_hidden_layers
num_attention_heads
vocab_size
max_position_embeddings
```

For BERT base, commonly:

```text
hidden_size = 768
num_hidden_layers = 12
num_attention_heads = 12
head dimension = 64
vocab_size ≈ 30522
max_position_embeddings ≈ 512
```

### `model.eval()`

Switches the model to evaluation mode.

For example, dropout behaves differently during evaluation.

```python
model.eval()
```

### `torch.no_grad()`

Disables gradient tracking during inference.

```python
with torch.no_grad():
    outputs = model(**inputs)
```

These are different:

```text
model.eval()
→ evaluation behavior

torch.no_grad()
→ don't track gradients
```

They are commonly used together for inference.

### CPU/GPU

Check GPU availability:

```python
import torch

print(torch.cuda.is_available())
```

Choose a device:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Move model:

```python
model.to(device)
```

Move inputs:

```python
inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}
```

The model and tensors must be on compatible devices.

### Parameter Count

```python
total_params = sum(
    parameter.numel()
    for parameter in model.parameters()
)
```

`numel()` counts the number of elements in a tensor.

### Precision

Common numerical types affect memory.

Conceptually:

```text
float32 → 4 bytes/value
float16 → 2 bytes/value
```

Actual deployment choices depend on the model, hardware, and workload.

### BERT Checkpoint Warning

When loading:

```python
AutoModel.from_pretrained("bert-base-uncased")
```

you may see warnings about unexpected keys such as:

```text
cls.predictions.*
cls.seq_relationship.*
```

These are associated with BERT's pretraining heads, while `AutoModel` loads the base encoder representation.

This can be expected and is not necessarily an error.

---

# 16. Inference Pipelines

The Hugging Face `pipeline()` API provides a high-level interface for inference.

It can automate:

```text
Input
 ↓
Tokenization
 ↓
Model
 ↓
Post-processing
 ↓
Result
```

### Sentiment Analysis

```python
from transformers import pipeline


classifier = pipeline(
    "sentiment-analysis"
)


result = classifier(
    "I love machine learning."
)

print(result)
```

### Fill-Mask

```python
fill_mask = pipeline(
    "fill-mask",
    model="bert-base-uncased"
)


result = fill_mask(
    "The capital of France is [MASK]."
)

print(result)
```

### Question Answering

```python
qa = pipeline(
    "question-answering"
)


context = (
    "Paris is the capital and most populous "
    "city of France."
)

question = "What is the capital of France?"


result = qa(
    question=question,
    context=context
)

print(result)
```

Extractive QA can return information such as:

```text
answer
score
start
end
```

The model identifies an answer span inside the provided context.

### Text Generation

```python
generator = pipeline(
    "text-generation",
    model="gpt2"
)


result = generator(
    "Artificial intelligence is",
    max_new_tokens=30
)

print(result)
```

### `max_new_tokens`

Controls the number of newly generated tokens rather than simply specifying the total input-plus-output sequence length.

### CPU/GPU

For CPU inference, a pipeline can use:

```python
device=-1
```

A GPU configuration can commonly use:

```python
device=0
```

when CUDA is available and the environment is configured appropriately.

### Why Use Pipelines?

Good for:

- Prototyping
- Quick experiments
- Simple inference
- High-level applications

Manual model/tokenizer APIs provide more control for:

- Debugging
- Custom preprocessing
- Custom model logic
- Detailed outputs
- Production-specific control

---

# 17. Fine-Tuning Concepts

### What is Fine-Tuning?

Fine-tuning means continuing the training of a pretrained model on a task-specific dataset.

```text
Pretrained model
      +
Task-specific dataset
      ↓
Fine-tuning
      ↓
Task-specific model
```

### Transfer Learning

Transfer learning means using knowledge learned from one training setting and adapting it to another task.

```text
Large-scale pretraining
        ↓
General knowledge
        ↓
Task-specific fine-tuning
```

### Pretraining vs Fine-Tuning

| Stage | Purpose |
|---|---|
| Pretraining | Learn general language patterns/representations |
| Fine-tuning | Adapt the model to a specific task |

### Classification Head

For sequence classification:

```text
Text
 ↓
Tokenizer
 ↓
BERT
 ↓
Contextual representation
 ↓
Classification Head
 ↓
Logits
 ↓
Class
```

Use:

```python
from transformers import AutoModelForSequenceClassification
```

Example:

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
```

### Forward Pass

```text
Input
 ↓
Model
 ↓
Prediction
```

For classification, the output includes logits.

Example:

```text
[-1.2, 2.7]
```

The values are raw scores, not probabilities.

### Loss

Loss measures how far the model's prediction is from the target.

```text
Prediction
     ↓
Compare with label
     ↓
Loss
```

Lower loss generally indicates better agreement with the training target, though validation metrics are essential for judging generalization.

### Backpropagation

```text
Loss
 ↓
Backward pass
 ↓
Gradients
 ↓
Optimizer
 ↓
Updated weights
```

### Parameter Update

Conceptually:

```text
new_weight
=
old_weight
-
learning_rate × gradient
```

### Epoch

One complete pass through the training dataset.

### Batch Size

Number of training examples processed in one batch.

### Learning Rate

Controls the size of parameter updates.

Too large:

```text
Training can become unstable
```

Too small:

```text
Training can become very slow
```

### Feature Extraction vs Fine-Tuning

Feature extraction:

```text
Pretrained model
      ↓
Frozen
      ↓
Features
      ↓
Train task head
```

Fine-tuning:

```text
Pretrained model
      ↓
Trainable
      ↓
Task head
```

### Freezing Layers

Example:

```python
for param in model.bert.parameters():
    param.requires_grad = False
```

This prevents those parameters from being updated through gradient-based training.

Useful when:

- Dataset is small
- Compute is limited
- You want to preserve pretrained representations

The decision should be evaluated rather than applied automatically.

### Forward-Pass Example

```python
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


model_name = "bert-base-uncased"


texts = [
    "I love this movie",
    "This movie was terrible",
    "Amazing experience",
    "I hated it"
]

labels = torch.tensor([1, 0, 1, 0])


tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)


encodings = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)


outputs = model(
    **encodings,
    labels=labels
)


print("Loss:")
print(outputs.loss)


print("\nLogits:")
print(outputs.logits)


predictions = torch.argmax(
    outputs.logits,
    dim=-1
)


print("\nPredictions:")
print(predictions)
```

### Important

The code above performs a forward pass and calculates loss.

It does **not** by itself perform complete fine-tuning.

Actual fine-tuning requires:

```text
Forward pass
 ↓
Loss
 ↓
loss.backward()
 ↓
optimizer.step()
 ↓
optimizer.zero_grad()
 ↓
Repeat over batches/epochs
```

### Optimizer Example

```python
from torch.optim import AdamW


optimizer = AdamW(
    model.parameters(),
    lr=2e-5
)
```

### Training Step

```python
outputs = model(
    **batch,
    labels=batch["labels"]
)

loss = outputs.loss

loss.backward()

optimizer.step()

optimizer.zero_grad()
```

### Overfitting

Fine-tuning can overfit.

A typical warning pattern is:

```text
Training loss ↓
Validation loss ↓ then ↑
```

This can indicate that the model is fitting the training data too closely.

Use training/validation/test splits and appropriate evaluation metrics.

### LoRA

LoRA (Low-Rank Adaptation) is a parameter-efficient adaptation method.

Instead of updating all pretrained parameters:

```text
Original weights → mostly frozen
Small trainable adapter parameters → learned
```

This can substantially reduce the number of trainable parameters and memory requirements.

It becomes especially useful for adapting larger models.

---

# 18. Context Length

### What Is Context?

Context is the sequence of tokens available to the model for understanding the current input and generating output.

### Context Window

The context window is the maximum amount of token context a model can process at once according to its architecture/configuration.

```text
┌──────────────────────────────┐
│       Context Window         │
│ token token token ...        │
└──────────────────────────────┘
```

### Context Is Measured in Tokens

Not words.

One word may correspond to:

```text
1 token
```

or multiple tokens depending on the tokenizer.

Therefore:

```text
Context limit = token limit
```

not a word limit.

### Why Is There a Limit?

Self-attention creates interactions across token positions.

For a sequence length `n`, the attention score matrix is approximately:

```text
n × n
```

Therefore increasing sequence length can significantly increase computation and memory requirements.

### Input + Output

For autoregressive generation, the input context and generated continuation must fit within the applicable context constraints.

Conceptually:

```text
Context Window
┌────────────────────────────────────┐
│ Input / Prompt │ Generated Output  │
└────────────────────────────────────┘
```

If the prompt is already very long, less room may remain for generation.

The exact behavior depends on the model and API.

### Truncation

Hugging Face tokenizers can truncate long sequences:

```python
encodings = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)
```

### `max_length`

Can explicitly limit the sequence length during tokenization:

```python
inputs = tokenizer(
    text,
    max_length=10,
    truncation=True,
    padding="max_length",
    return_tensors="pt"
)
```

### Padding

Makes sequences reach a common length.

### Truncation

Removes tokens beyond the selected maximum.

```text
Padding
→ increases shorter sequences

Truncation
→ reduces longer sequences
```

### Attention Mask

For padded input:

```text
1 → real token
0 → padding
```

### Context Length vs Attention Mask

Context length:

```text
Maximum processing window
```

Attention mask:

```text
Controls which positions are considered valid/visible under the relevant masking scheme
```

They are different concepts.

### Long Documents

A long document may not fit in one model input.

Possible strategies include:

```text
Long document
 ↓
Chunking
 ↓
Process chunks
```

For retrieval-based systems:

```text
Document
 ↓
Chunking
 ↓
Embeddings
 ↓
Vector database
 ↓
Retrieve relevant chunks
 ↓
Question + relevant context
 ↓
LLM
```

This is an important reason RAG is useful.

### Context Quality

A larger context window does not automatically mean better answers.

Relevant, high-quality context is often more useful than large amounts of irrelevant information.

Consider:

```text
Question
+
5 relevant chunks
```

versus:

```text
Question
+
thousands of irrelevant tokens
```

The second may be technically possible but is not necessarily a better engineering solution.

---

# 19. Generation Parameters

Text generation is an autoregressive process.

```text
Prompt
 ↓
Transformer
 ↓
Logits
 ↓
Probability distribution
 ↓
Choose next token
 ↓
Append token
 ↓
Repeat
```

### Logits

The model produces raw scores for possible next tokens.

Example:

```text
technology → 2.8
powerful   → 2.4
changing   → 1.9
banana     → -2.1
```

These are logits.

Softmax converts them into probabilities.

---

# 20. Greedy Decoding

Greedy decoding selects the highest-probability next token.

```text
Highest probability
       ↓
Select token
```

This is deterministic for a fixed model/input and decoding setup.

It can be predictable.

---

# 21. Sampling

Sampling selects tokens according to the probability distribution.

Instead of always selecting the highest-probability token, lower-probability candidates may sometimes be selected.

Useful for:

- Creative writing
- Brainstorming
- Diverse generation
- Conversational variation

Less useful when deterministic behavior is preferred.

---

# 22. `do_sample`

```python
do_sample=False
```

generally uses deterministic decoding such as greedy decoding when no beam-search strategy changes the decoding behavior.

```python
do_sample=True
```

enables sampling-based generation.

Example:

```python
generator(
    prompt,
    max_new_tokens=30,
    do_sample=True
)
```

---

# 23. `max_new_tokens`

Controls approximately how many new tokens can be generated.

Example:

```python
max_new_tokens=30
```

means up to about 30 newly generated tokens, subject to the model's context and generation constraints.

It is not the same as saying the entire input + output must be 30 tokens.

---

# 24. `max_length`

Generation APIs can also expose:

```python
max_length=50
```

Historically this represents a maximum sequence length for generation and can interact with input length.

When the intent is specifically:

> Generate N additional tokens

`max_new_tokens` is usually clearer.

---

# 25. Temperature

Temperature modifies the logits before softmax.

Conceptually:

```text
P_i =
exp(z_i / T)
/
Σ exp(z_j / T)
```

where:

- `z_i` = logit
- `T` = temperature

### Lower Temperature

```text
More concentrated distribution
↓
More predictable choices
```

### Higher Temperature

```text
Flatter distribution
↓
More diversity / unpredictability
```

Temperature changes the sampling distribution. It does not make the model inherently more intelligent.

---

# 26. Top-k Sampling

`top_k` limits sampling to the `k` highest-probability candidate tokens.

Example:

```python
top_k=5
```

means:

```text
All vocabulary tokens
       ↓
5 highest-probability tokens
       ↓
Sampling
```

Example:

```text
A = 0.40
B = 0.25
C = 0.15
D = 0.08
E = 0.05
F = 0.03
...
```

With:

```text
top_k=3
```

candidate tokens are:

```text
A
B
C
```

---

# 27. Top-p Sampling

Top-p is also called nucleus sampling.

Instead of a fixed number of tokens, it selects the smallest set of highest-probability tokens whose cumulative probability reaches the threshold `p`.

Example:

```text
A = 0.50
B = 0.25
C = 0.12
D = 0.06
E = 0.04
F = 0.03
```

With:

```text
top_p=0.90
```

cumulative probabilities:

```text
A              = 0.50
A + B          = 0.75
A + B + C      = 0.87
A + B + C + D  = 0.93
```

Therefore A, B, C, and D are sufficient to reach the threshold.

### Top-k vs Top-p

```text
top_k
→ fixed number of candidates

top_p
→ probability-based variable-size candidate set
```

---

# 28. Temperature + Top-p

Example:

```python
generator(
    prompt,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)
```

Conceptually:

```text
Logits
 ↓
Temperature adjustment
 ↓
Probability distribution
 ↓
Top-p filtering
 ↓
Sampling
 ↓
Next token
```

The exact generation implementation determines the precise processing details.

---

# 29. Repetition Penalty

A repetition penalty discourages repeated tokens.

Example:

```python
repetition_penalty=1.1
```

Conceptually:

```text
Previously generated token
        ↓
Penalty
        ↓
Lower preference for repetition
```

Values above 1 generally discourage repetition.

Excessive penalties can make output unnatural.

---

# 30. Beam Search

Beam search keeps multiple candidate sequences instead of following only one.

Example:

```text
Start
 ├── A
 ├── B
 └── C
```

If:

```text
num_beams=3
```

the algorithm keeps several candidate sequences and expands/scorers them to search for high-scoring sequences.

### Greedy vs Beam Search

Greedy:

```text
Choose best next token
 ↓
Continue
```

Beam:

```text
Keep multiple candidate sequences
 ↓
Expand candidates
 ↓
Score
 ↓
Keep strongest candidates
```

Beam search is useful for some sequence-level decoding tasks, especially some encoder-decoder applications.

### Beam Search vs Sampling

```text
Beam search
→ search among multiple high-scoring candidate sequences

Sampling
→ probabilistically select from candidate tokens
```

They serve different decoding goals.

---

# 31. Practical Generation Example

```python
from transformers import pipeline


generator = pipeline(
    "text-generation",
    model="gpt2"
)


prompt = "Artificial intelligence is"


result = generator(
    prompt,
    max_new_tokens=30,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)


print(result[0]["generated_text"])
```

### Explanation

```python
pipeline("text-generation", model="gpt2")
```

creates a text-generation pipeline using GPT-2.

```python
max_new_tokens=30
```

limits newly generated tokens.

```python
do_sample=True
```

enables probabilistic sampling.

```python
temperature=0.7
```

makes the sampling distribution more focused.

```python
top_p=0.9
```

restricts sampling to a nucleus of high-probability tokens.

---

# 32. Generation Parameter Summary

| Parameter | Purpose |
|---|---|
| `max_new_tokens` | Controls number of newly generated tokens |
| `max_length` | Maximum generation sequence length behavior |
| `do_sample` | Enables sampling |
| `temperature` | Controls probability-distribution sharpness |
| `top_k` | Limits candidates to top k tokens |
| `top_p` | Limits candidates by cumulative probability |
| `repetition_penalty` | Discourages repetition |
| `num_beams` | Controls beam-search width |

### Mental Model

```text
                 PROMPT
                    ↓
               Transformer
                    ↓
                  LOGITS
                    ↓
              Temperature
                    ↓
             Probabilities
                    ↓
              Top-k / Top-p
                    ↓
            Sampling / Search
                    ↓
               Next Token
                    ↓
                  Repeat
```

---

# 33. Common Mistakes

### Temperature

Wrong:

```text
Temperature = intelligence
```

Correct:

```text
Temperature = sampling distribution control
```

### Top-k

Wrong:

```text
top_k=10 → 10 words
```

Correct:

```text
top_k=10 → 10 token candidates
```

### Top-p

Wrong:

```text
top_p=0.9 → 90% of tokens
```

Correct:

```text
top_p=0.9 → candidate set reaching
            approximately 90% cumulative probability
```

### `max_new_tokens`

Wrong:

```text
max_new_tokens=100
→ total sequence is 100
```

Correct:

```text
→ approximately up to 100 newly generated tokens
```

---

# 34. Complete Transformer Mental Model

```text
                         RAW TEXT
                            │
                            ▼
                       TOKENIZER
                            │
                            ▼
                        TOKEN IDs
                            │
                            ▼
                     TOKEN EMBEDDINGS
                            │
                            ▼
                 POSITIONAL INFORMATION
                            │
                            ▼
                  ┌───────────────────┐
                  │    TRANSFORMER    │
                  │                   │
                  │ Attention         │
                  │ Q / K / V         │
                  │ Multi-Head        │
                  │ FFN               │
                  │ Residual          │
                  │ Normalization     │
                  └─────────┬─────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
            BERT           GPT          Llama
          Encoder-only   Decoder-only  Decoder-only
              │             │             │
              ▼             ▼             ▼
        Understanding   Generation   Generation
```

---

# 35. Transformer Module Progress

```text
✅ 01. Sequence-to-sequence problem
✅ 02. Attention intuition
✅ 03. Query, Key, Value
✅ 04. Self-attention
✅ 05. Multi-head attention
✅ 06. Positional information
✅ 07. Encoder vs decoder
✅ 08. Transformer block
✅ 09. BERT + Masked Language Modeling
✅ 10. GPT-style decoder-only architecture
✅ 11. Llama overview
✅ 12. Tokenization + subword tokens
✅ 13. Hugging Face Transformers
✅ 14. Loading pretrained models
✅ 15. Inference pipelines
✅ 16. Fine-tuning concepts
✅ 17. Context length
✅ 18. Generation parameters

NEXT:
⬜ 19. QA System
⬜ 20. Evaluation
⬜ 21. FastAPI / simple UI
⬜ 22. Docker
⬜ 23. Deployment
```

## Quick Revision Checklist

Before moving to the QA project, you should be able to explain:

- What problem Seq2Seq solves
- Why attention was introduced
- Q, K, V and scaled dot-product attention
- Why `√dk` is used
- Self-attention
- Multi-head attention
- Why positional information is needed
- Encoder vs decoder
- Causal masking
- Cross-attention
- Transformer block
- Residual connections
- LayerNorm
- FFN
- BERT and MLM
- GPT-style decoder-only architecture
- Llama-style improvements
- Tokenization and subword tokens
- Padding vs truncation
- Attention mask vs causal mask
- Hugging Face tokenizers/models/pipelines
- `AutoModel` vs task-specific model classes
- Pretrained models
- `model.eval()` vs `torch.no_grad()`
- Fine-tuning
- Feature extraction vs fine-tuning
- Forward pass, loss, backpropagation, optimizer
- Epoch, batch size, learning rate
- Context length
- Chunking for long documents
- Greedy decoding
- Sampling
- Temperature
- Top-k
- Top-p
- Repetition penalty
- Beam search
- `max_new_tokens`

This file intentionally covers the complete Transformer material we've studied from **Sequence-to-Sequence through Generation Parameters**, preserving the roadmap topics without skipping any of them.
