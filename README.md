# modern-hopfield-context-retrieval
Modular PyTorch implementation of context retrieval using modern Hopfield networks, cross-attention, and similarity modules.


The structure:

modern-hopfield-context-retrieval
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── models
│   ├── hopfield
│   │   └── my_hopfield.py
│   │
│   ├── context_module.py
│   ├── cross_attention_module.py
│   └── similarity_module.py
│
├── tests
│   ├── test_hopfield.py
│   ├── test_context_module.py
│   └── test_similarity_module.py
│
├── examples
│   └── example_usage.ipynb
│
└── docs
    └── architecture.md


Modern Hopfield Context Retrieval

PyTorch implementation of a modular context-aware retrieval architecture using modern Hopfield networks, cross-attention, and similarity-based reasoning.

This repository provides reusable neural modules for associative memory and context-based inference.

Architecture

The system consists of the following components:

Context Module

Aggregates information from a support set and enriches query representations.

Cross-Attention Module

Performs attention-based interaction between queries and contextual information.

Similarity Module

Computes similarity-based retrieval scores between query and support embeddings.

Hopfield Memory

Implements a modern Hopfield network for associative memory retrieval.
