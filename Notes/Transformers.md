# Transformers

* text -> tokens -> vector (each word is converted to a list of words)
* With attention each vector values is updated. like the meaning of each word with respect to other words.
* After that these vectors passed on to MultiLayered perceptron/ Feed farward layer
    * Each vectors doesn't interact but every vectors processed parallely
* The above process repeated multiple times.

* Parameters are like slope and y-intercept in linear regression but in LLMs it is like in Billions.
* Ex: the 175 Billions parameters in GPT 3 comes from organized 28K matrices
* Embedding, Key, Query, Value, Output, up-projection, down-projection, unembedding

# Text Embeddings — From Classic ML to GPT (Clean Intuition Notes)

## Big Picture: What an Embedding Is

* An **embedding** is a **lookup table** that maps symbols (words / tokens) to **dense numeric vectors**.
* These vectors are what neural networks actually process.

> **Embeddings turn language into geometry.**

---

## Two Fundamental Questions (Every System Answers These)

Every text system must answer:

1. **How do I turn text into numbers?**
2. **Where do those numbers come from (training vs prediction)?**

All embedding methods differ only in **how** they answer these two questions.

---

## Vocabulary and Embedding Space

* Embedding tables contain:

  * A list of **all vocabulary tokens**
  * A corresponding **vector for each token**

Example shape:

```
(vocab_size × embedding_dim)
```

* Word embeddings are **high dimensional**

  * GPT‑3 hidden size ≈ **12,288 dimensions**
* High-dimensional space cannot be visualized directly

### Visual intuition

* Think of embeddings living in a **very high-dimensional space**
* We can only visualize them by:

  * Taking a 2D or 3D slice
  * Projecting vectors down (PCA, t‑SNE)

---

## Scale Example (GPT‑3)

* Vocabulary size ≈ **50,000** tokens
* Embedding dimension ≈ **12,288**

Total embedding parameters:

```
50,000 × 12,288 ≈ 617 million weights
```

This is **just the embedding table**, before attention or MLP.

---

## Types of Embedding / Representation

---

## 1️⃣ Bag of Words (BoW) — Classic ML

### How text becomes numbers

* Build vocabulary from training corpus (**fixed forever**)
* Vector length = vocabulary size
* Initialize all zeros

Example:

```
"I love ML" → [0, 1, 1, 1]
```

### Key properties

* No learning of representation
* Counts **are** the representation
* Word order is ignored

### New input behavior

* Words not in vocabulary → **ignored**

---

## 2️⃣ TF‑IDF — Weighted BoW

### How text becomes numbers

* Same pipeline as BoW
* Values are **weighted**, not raw counts

```
TF‑IDF = TF × log(total_docs / docs_with_word)
```
### Term Frequency (TF)
$$
\mathrm{TF}(t, d) =
\frac{\text{Number of times term } t \text{ appears in document } d}
{\text{Total number of terms in document } d}
$$
### Inverse Document Frequency (IDF)
$$
\mathrm{IDF}(t, D) =
\log\left(
\frac{\text{Total number of documents in corpus } D}
{\text{Number of documents containing term } t}
\right)
$$

### Intuition

* Rare, informative words → higher values
* Common words → lower values

### Output form (important)

> **One sentence → one fixed-length vector**

Same shape as BoW, different values.

---

### Summary (BoW & TF‑IDF)

> In BoW and TF‑IDF, a new input sentence is converted into a **fixed-length vector equal to the vocabulary size**, where each position corresponds to a vocabulary word and contains either a **count (BoW)** or a **TF‑IDF weight (TF‑IDF)**.

---

## 3️⃣ Embedding Matrix — Neural Networks Begin

This is the first time **representations are learned**.

### Training phase

1. Build vocabulary
2. Convert words → token IDs
3. Initialize embedding matrix **randomly**

```
(vocab_size × embedding_dim)
```

4. During training, **backpropagation updates embeddings**

> Words used in similar contexts move closer in vector space.

---

### Using embeddings in LSTM (example)

#### Training

```
text → ids → embeddings → LSTM → loss
                    ↑
              embeddings learn
```

* Embeddings encode **semantics**
* LSTM composes meaning **sequentially**

#### Prediction (inference)

```
text → ids → trained embeddings → LSTM → output
```

* No learning
* No weight updates
* Only forward pass

---

## 4️⃣ Word2Vec and GloVe — Pretrained Embeddings

### What they are

* Models whose **main goal** is to learn good word embeddings
* After training:

  * ❌ discard the model
  * ✅ keep the embedding matrix

Result:

```
word → vector lookup table
```

---

### Word2Vec details

* Word2Vec **is a neural network**

#### Skip‑gram (most common)

* **Goal**: Given a center word → predict surrounding words

Architecture:

```
one‑hot → embedding matrix → vocab‑size softmax
```

* Output layer represents **words**, not embeddings
* The hidden‑layer weights **are the embeddings**

#### CBOW

* Given surrounding words → predict center word

---

### Key takeaway

> **Word2Vec is trained to predict words, but we only keep the hidden‑layer weights as embeddings.**

---

## 5️⃣ Images vs Text (Important Contrast)

* **Text must be converted into numbers first**
* **Images are already numbers**, but poorly structured

Embeddings play the same role:

| Text             | Vision         |
| ---------------- | -------------- |
| Word / token     | Pixel / patch  |
| Embedding vector | Feature vector |

---

## 6️⃣ Embeddings in Transformers and GPT

Transformers generalize embeddings and make them **contextual**.

### Full pipeline

```
Text
 → Sub‑word tokenization
 → Integer IDs
 → Token embedding lookup
 → + Positional embeddings
 → Self‑Attention (context mixing)
 → Feed‑Forward Network (vector refinement)
 → Layer stacking
```

---

### Key difference from Word2Vec / GloVe

* Static embeddings:

  * Same word → same vector

* Transformer embeddings:

  * Same token → **different vector in different contexts**

This is called **contextual embedding**.

---

## Final Mental Model (Lock This In)

> **Classic ML**: text → statistics
> **Neural nets**: text → learned vectors
> **Transformers/GPT**: text → vectors reshaped by context

Embeddings are the foundation of all modern language models.

---



# Context Size & Softmax — Clean Intuition Notes

## Context Size (Context Window)

### What Context Size Means

* **Context size** is the maximum number of tokens a Transformer model can *see at once* when predicting the next token.
* It defines how much past text can influence the current prediction.

> **Context size = model’s short‑term memory**

---

### Important Clarification

* Context size does **not** mean knowledge size
* Facts are stored in **model weights**, not in context

> Context only determines **what information is available right now**, not what the model knows.

---

## Softmax

### What Softmax Does

* **Softmax** converts an arbitrary list of real numbers (called **logits**) into a **probability distribution**.
* Output values:

  * Are between 0 and 1
  * Sum to 1

> **Softmax turns scores into probabilities**

---

### Mathematical Definition

For logits `z₁, z₂, ..., zₙ`:

```
softmax(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)
```

* Larger logits → higher probabilities
* Smaller logits → probabilities close to 0

---

### Intuition

* Softmax **amplifies differences** between values
* The largest logit dominates the distribution

Example:

```
[2.0, 1.0, 0.1] → [0.66, 0.24, 0.10]
```

---

## Temperature (T)

### Why Temperature Exists

Temperature controls **how sharp or flat** the softmax distribution is.

Softmax with temperature:

```
softmax(zᵢ / T)
```

---

### Effect of Temperature Values

#### Low Temperature (T < 1)

* Makes the distribution **sharper**
* High‑probability tokens become **more dominant**
* Model behaves more **deterministically**

> Useful when you want safe or predictable outputs

---

#### High Temperature (T > 1)

* Makes the distribution **flatter**
* Lower‑probability tokens get **more chance**
* Model becomes more **creative or random**

> Useful for creative text generation

---

#### Edge Case: T → 0

* Softmax collapses to **argmax**
* Always picks the highest‑logit token

> No randomness at all

---

### One‑Line Mental Model

> **Softmax decides probabilities.**
> **Temperature decides how confident the model is about its choice.**


# Transformer Attention — Clean Intuition Notes

## Big Picture (Why Attention Exists)

* **Word embeddings alone are static**
  They come from a lookup table and **do not know context**.

  * `"tower"` → same vector everywhere

* **Attention adds context**

  * `"tower"` → *French* → *miniature* → *gift*
  * Same word, different meaning → different vector **after attention**

> **Attention = context injection mechanism**

---

## Where Attention Fits in a Transformer

A Transformer block consists of:

1. **Self-Attention** (≈ 1/3 of parameters)
2. **Feed Forward Network (MLP)**
3. **Residual Connections + LayerNorm**

These notes focus **only on Self-Attention**.

---

## Example Sentence

> `"a fluffy blue creature roamed the verdant forest"`

After tokenization + embedding + positional encoding:

```
E1   E2     E3    E4        E5     E6      E7      E8
a | fluffy | blue | creature | roamed | the | verdant | forest
```

Each `Ei` is a vector (e.g., 768-dimensional).

---

## What a Single-Head Self-Attention Does

### Step 1: Start with Position-Aware Embeddings

Each token embedding already includes position information:

```
E1, E2, E3, ..., E8
```

---

### Step 2: Create Query, Key, Value vectors

Each embedding is projected in **three different ways**:

```
Qi = Wq · Ei
Ki = Wk · Ei
Vi = Wv · Ei
```

Where:

* `Wq`, `Wk`, `Wv` are **learned weight matrices**
* They reshape / project embeddings into different subspaces

Result:

```
Q1 Q2 Q3 ... Q8
K1 K2 K3 ... K8
V1 V2 V3 ... V8
```

**Intuition**:

* **Query** → What am I looking for?
* **Key** → What do I contain?
* **Value** → What information do I provide?

### Concrete Example for Q, K, V (Intuition)

Using the sentence:

> `"a fluffy blue creature roamed the verdant forest"`

Focus on the token **"creature"**.

#### Query (Q) — *What am I looking for?*

* The Query vector of **"creature"** represents the *kind of information it wants*.
* Intuitively, it is asking:

> “Am I described by something?”
> “Is there an adjective or attribute related to me?”

So `Q(creature)` is tuned to match:

* adjectives
* descriptors
* modifiers

---

#### Key (K) — *What do I contain?*

Each surrounding word advertises what it has:

* `K(fluffy)` → “I am a descriptive adjective.”
* `K(blue)` → “I describe color.”
* `K(roamed)` → “I am an action.”

Keys act like **labels or metadata** saying:

> “This is the type of information I carry.”

---

#### Query × Key — *Who should I listen to?*

* `Q(creature) · K(fluffy)` → high score ✅
* `Q(creature) · K(blue)` → high score ✅
* `Q(creature) · K(roamed)` → low score ❌

This step decides:

> **Which words are relevant to "creature"**

But this still produces **only numbers**, not meaning.

---

#### Value (V) — *What information do I actually provide?*

Now comes the crucial part.

* `V(fluffy)` contains **softness / texture features**
* `V(blue)` contains **color features**
* `V(roamed)` contains **movement/action features**

These are the **actual semantic contents**.

---

#### Final Mixing — *Context injection*

The attention weights (from QK) are used to mix Values:

```
creature_new = 0.45·V(fluffy) + 0.40·V(blue) + 0.05·V(roamed)
```

So the new representation becomes:

> **"fluffy blue creature"**

---

### One-line takeaway (lock this in)

> **Q asks the question, K decides relevance, and V supplies the meaning.**

---

### Step 3: Compute similarity (attention scores)

For each Query, compare it with **all Keys**.

For example, for `Q4` ("creature"):

```
score(Q4, Ki) = (Q4 · Ki) / √dk
```

Why divide by `√dk`?

* Prevents dot products from exploding
* Keeps softmax stable

This produces:

```
[Q4·K1, Q4·K2, ..., Q4·K8]
```

---

### Step 4: Softmax → Attention weights

Apply softmax to get probabilities:

```
α4 = softmax([Q4·K1, ..., Q4·K8])
```

Properties:

* Values sum to 1
* Larger value = stronger attention

**Example intuition**:

* `"creature"` attends strongly to `"fluffy"` and `"blue"`
* Weakly attends to `"the"` or `"forest"`

---

### Step 5: Weighted sum of Values (context mixing)

Use attention weights on Value vectors:

```
Output4 = α41·V1 + α42·V2 + ... + α48·V8
```

This creates a **contextualized vector** for `"creature"`:

> `"fluffy blue creature"`

This is where **context is injected**.

---

## 🔍 What exactly is **V (Value)** and why do we need it?

This is a **very important conceptual point**, and your confusion is natural.

### Key idea (one line)

> **Q and K decide *who to listen to*.
> V decides *what information is actually taken*.**

---

### Why Q × K alone is NOT enough

* The dot product **Q · K** only produces **attention weights** (numbers).
* These weights answer only:

> *“How relevant is token j to token i?”*

But relevance alone does **not carry information**.

Example:

* Q–K can tell that `"creature"` should attend to `"fluffy"`
* But it does **not say what to extract from `"fluffy"`**

So we still need **content vectors**.

---

### What V actually represents

`V = Wv · E`

* `V` is a **transformed version of the original embedding**
* It encodes **what information this token offers to others**

Think of it as:

* adjectives → descriptive features
* nouns → entity features
* verbs → action features

---

### Concrete intuition (your sentence)

For the word `"fluffy"`:

* **K** says: *“I am an adjective, relevant to nouns.”*
* **V** says: *“Here is softness / texture information.”*

For the word `"blue"`:

* **K** says: *“I modify visual attributes.”*
* **V** says: *“Here is color information.”*

When `"creature"` attends to them:

* Q–K decides **which words matter**
* Attention weights decide **how much they matter**
* **V provides the actual features that get mixed in**

---

### Why we cannot directly use embeddings instead of V

If we used raw embeddings instead of `V`:

* The same information would be reused for **every attention head**
* No specialization would happen

`Wv` allows each attention head to:

* Focus on different aspects
* Extract different kinds of information

This is critical for **multi-head attention**.

---

### One-sentence summary (lock this in)

> **Q and K compute the attention pattern.
> V carries the information that gets blended using that pattern.**

---

## Attention Pattern (Matrix View)

Doing this for all tokens forms an **attention matrix**:

* Rows → Query positions
* Columns → Key positions

```
Q1 → [K1 K2 K3 ... K8]
Q2 → [K1 K2 K3 ... K8]
...
Q8 → [K1 K2 K3 ... K8]
```

This matrix is called the **attention pattern**.

---

## GPT-Specific Detail: Masked Self-Attention

In GPT (decoder-only Transformer):

* Tokens **cannot attend to future tokens**
* Future positions are masked

Mechanism:

* Masked scores → `-∞`
* After softmax → probability becomes `0`

Result:

* Attention matrix is **lower triangular**
* Upper-right half is zero

This enforces **causal (autoregressive) behavior**.

---

Correct pipeline:

```
E → Q, K, V → attention weights → weighted V sum → contextual vector
```

---

## Residual Connection (Context Accumulation)

The attention output is **added back** to the original embedding:

```
E_new = E + Attention(E)
```

This ensures:

* Original meaning is preserved
* Context is layered gradually

---

## Multi-Head Attention (Quick intuition)

* Single head → one relationship type
* Multiple heads → multiple perspectives

Examples:

* Head 1 → adjective–noun
* Head 2 → subject–verb
* Head 3 → long-range dependency

All heads are concatenated and projected.

---

## Cross-Attention (Translation case)

Used in encoder–decoder models:

* **Keys + Values** → source language
* **Queries** → target language

Allows words in one language to attend to relevant words in another.

(Not used in GPT, but conceptually important.)

---

## Final Mental Model

> **Self-attention lets each token ask:**
> **“Which other tokens should I listen to in order to understand myself better?”**



# Multilayered Perceptron (2/3 of parameters here)
# Transformer MLP (Feed-Forward Network) — Clean Intuition Notes

## Big Picture (Why MLP Exists)

* In a Transformer, **attention mixes information across tokens**.
* But after mixing, each token still needs to **think independently**.

> **Attention = communication between tokens**
> **MLP = private thinking of each token**

This is why the **MLP holds ~2/3 of the parameters**.

---

## What the MLP Is (High-level)

The Transformer MLP is also called the:

* **Position-wise Feed Forward Network (FFN)**

Key property:

> **Each token is processed independently and identically.**

There is **NO interaction between tokens** inside the MLP.

---

### Core idea

* The *facts* and *knowledge* of the model live mainly in the **MLP weights**.
* Each token vector undergoes a **series of transformations**.
* The output stays the **same dimension** as the input.
* The transformed vector is **added back** to the original token vector.

This is done **for every token independently**.

---

## Exact Computation (Per Token)

For one token vector `x` (e.g., 768-dim):

```
MLP(x) = W2 · ReLU(W1 · x + b1) + b2
```

Then the residual connection:

```
x_new = x + MLP(x)
```

This same computation is applied to:

```
E1, E2, E3, ..., En
```

---

## Step-by-Step Intuition

### Step 1: Linear Projection (W1)

```
h = W1 · x + b1
```

* Expands the dimension (e.g., 768 → 3072)
* Allows the model to ask **many questions in parallel**

Intuition:

> “Is this token expressing concept A?”
> “Is this token expressing concept B?”

---

### Step 2: ReLU (or GELU)

```
h' = ReLU(h)
```

* Acts like a **soft yes/no gate**
* Keeps useful activations
* Suppresses irrelevant ones

Intuition:

> “Only keep the concepts that apply here.”

---

### Step 3: Linear Projection Back (W2)

```
out = W2 · h' + b2
```

* Compresses back to original dimension (3072 → 768)
* Combines detected features into a refined representation

Intuition:

> “Summarize what I learned about this token.”

---

### Step 4: Residual Addition

```
x_new = x + out
```

* Preserves original meaning
* Adds refined information
* Enables stable deep training

---

## Concrete Token-Level Example

Consider the token:

> **"creature"** (after attention)

### What attention already did

* Mixed information from:

  * "fluffy"
  * "blue"

So now `x(creature)` roughly means:

> *"fluffy blue creature"*

---

### What the MLP now does

The MLP asks questions like:

* Is this a **living entity**?
* Is it **physical**?
* Is it **describable by attributes**?
* Is it **animate**?

Each neuron in `W1` corresponds to such a learned question.

ReLU activates only the relevant ones.

Then `W2` recombines them into a **cleaner semantic encoding**.

---

## Why MLP Contains Most "Facts"

* Attention decides **where to look**
* MLP decides **what to compute**

Facts like:

* "Paris is a city"
* "Creatures can roam"
* "Blue is a color"

Are encoded in the **weights of W1 and W2**, not in attention.

---

## Important Clarifications

### ❌ MLP does NOT mix tokens

* No cross-token interaction
* No context exchange

Context mixing happens **only in attention**.

---

### ✅ MLP is applied position-wise

Same network:

* Same weights
* Applied independently
* To every token

This is why it is called **position-wise FFN**.

---

## One-Line Mental Model (Lock This In)

> **Attention lets tokens talk to each other.
> MLP lets each token think on its own.**

---

## Ultra-Short Summary

* Attention = communication
* MLP = reasoning
* Attention moves information
* MLP transforms information
* Most parameters live in MLP

---

Intuition

Think of it like reading:
* Encoder (BERT) → can look left and right
* Decoder (GPT) → can only look left
* Encoder–decoder → decoder looks left + consults encoder notes
* But in all cases: Self-attention is always the core mechanism.

Bottom line
* GPT is not special for having self-attention.
* All Transformer decoders use masked self-attention.
* GPT differs because it removes cross-attention and the encoder.
* Understanding in GPT emerges entirely from stacked masked self-attention.
