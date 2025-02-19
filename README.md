# Paged Attention for GPT-2

Sequence generation in models like GPT-2 follows an autoregressive process, i.e., the model generates one token at a time. Each new token is conditioned on the previously generated tokens. This autoregressive process can be computationally expensive and slow, especially when handling long sequences.

Paged attention is a technique designed to accelerate the generation process in models like GPT-2. It breaks the attention mechanism into smaller chunks or pages, enabling the model to attend to only a subset of tokens at a time. This strategy significantly reduces the computational cost of the attention mechanism and speeds up the overall generation process.

## How it works?

### 1. Tokenization

The input text is first tokenized into individual tokens.

### 2. Page Generation

Each token is then converted into a dense vector embedding—a numerical representation that the model can process.

### 3. Self-Attention Layers

The embeddings are processed through multiple self-attention layers to compute the attention scores, which indicate the importance of each token in the context of the others. This is achieved by computing the following matrices:

- **Query Matrix (Q)**: Represents the current token’s request for context.
- **Key Matrix (K)**: Represents the significance of each token.
- **Value Matrix (V)**: Represents the content of each token.

Rather than computing the attention scores for every token in the sequence at every step, the model uses a query-key-value cache to store the attention data for tokens that have already been processed. This cache is updated incrementally as new tokens are generated.

### 4. Page Attention

- **Standard Caching**: In the traditional approach, the model stores the attention scores for all tokens in a single large tensor that grows with the sequence length. This method works well for short sequences but can become computationally expensive as the sequence length increases.

- **Paged Attention**: With paged attention, the key-value cache is divided into smaller, fixed-size pages rather than one large tensor. When new tokens are generated:

  - Their corresponding key and value matrices are stored in the current page.
  - If the current page is full, a new page is allocated.

This approach ensures that the model only processes a manageable subset of tokens at a time, reducing both memory overhead and computational cost.

### 5. Attention Computation and Output Generation

After the key and value vectors are cached (potentially across multiple pages), the model computes the attention scores for the current token:

- The query vector of the current token is compared against all cached keys (by concatenating the pages as needed).
- The resulting attention scores are scaled and normalized using a softmax function to obtain attention weights.
- These weights are then applied to the cached values to generate a context vector.
- The context vector is used to predict the next token, which is appended to the sequence.
- The key and value caches are updated with the new token’s information.

## Benefits of Paged Attention

- By splitting the key-value cache into fixed-size pages, the model avoids allocating one enormous contiguous tensor, which is particularly beneficial for long sequences.

- Processing a limited number of tokens at each step leads to significant reductions in computational cost.

- Paged attention scales better for long sequence generation, making it an attractive option for large autoregressive models like GPT-2.
