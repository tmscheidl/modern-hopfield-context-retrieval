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
