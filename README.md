# Natural Language Processing (NLP) Course Assignments

This repository contains the coursework assignments for DS 207 - Introduction to NLP at IISc. The assignments cover fundamental concepts and advanced techniques in Natural Language Processing.

**Student:** Abhishek Kumawat 

## Repository Structure

**Note:** This repository follows a .gitignore policy that excludes model files (.pt), datasets (.csv), checkpoints, compiled files (.py, .json), and other artifacts to keep the repository clean and focused on source notebooks.

```
nlp/
├── nlp_Assignment1/                            # Text Classification
│   ├── Assignment_1_for_DS_207_(Intro_to_NLP)_Text_Classification.ipynb
│   └── Abhishek_kumawat_24401/
│       └── Abhishek_Kumawat_24401_Assignment1.py
│
├── nlp_assignment2/                            # Language Modelling
│   ├── Assignment_2_for_DS_207_2025_(Intro_to_NLP)_Language_Modelling.ipynb
│   ├── Abhishek Kumawat_24401_assignment2.py
│   └── Abhishek Kumawat_24401/
│       └── Abhishek_kumawat_24401_assignment2/
│           └── Abhishek_kumawat_24401_assignment2.py
│
├── nlp_assignment3/                            # Sequence-to-Sequence Modeling  
│   └── Assignment_3_for_DS_207_2025(Intro_to_NLP)_Sequence_to_Sequence_Modeling.ipynb
│
└── nlp_Assignment4/                            # Transformer Analysis & Pruning
    ├── Assignment_4_for_DS_207_2025_(Intro_to_NLP).ipynb
    ├── head_imp_cnn.png                       # Attention head importance visualizations
    ├── head_imp_wmt.png
    ├── prune_any.png                          # Pruning analysis results
    └── prune_encoder_decoder_cross.png
```

### Files Excluded by .gitignore
The following types of files are excluded from version control:
- **Model files:** `*.pt`, `*.pkl`, `*.h5` (trained model weights)
- **Data files:** `*.csv`, `*.txt` (datasets and outputs)
- **Generated files:** `*.py` (most exported Python files), `*.json` (logs and metadata)
- **Archives:** `*.zip`, `*.tar`, `*.gz` (submission packages)
- **Images:** `*.png`, `*.pdf` (most visualizations and documentation)
- **Checkpoints:** `checkpoint*/`, `checkpoints/` (training intermediate states)
- **System files:** `.DS_Store`, `*.log`, `__pycache__/`, `.ipynb_checkpoints/`

## Assignment Overview

### Assignment 1: Text Classification and Word Vectors
**Objective:** Introduction to text classification and word embeddings

**Key Components:**
- **Generative Classification:** Naive Bayes classifier implementation using Bayes' theorem
- **Word2Vec and Word Analogies:** Working with pre-trained word embeddings and solving word analogies
- **Discriminative Classification:** Logistic regression for text classification
- **Dataset:** Sentiment analysis on positive/negative text reviews

**Technologies Used:** Python, NumPy, Pandas, NLTK, Scikit-learn, Gensim

### Assignment 2: Language Modelling
**Objective:** Character-level language modeling for city names generation

**Key Components:**
- **Statistical Language Models:** Unigram, Bigram, and Trigram models
- **Neural N-gram Language Model:** Feed-forward neural network for character prediction
- **RNN Language Model:** Recurrent Neural Network for sequence modeling
- **Model Training:** PyTorch implementation with checkpointing (models excluded from repo)
- **Evaluation:** Perplexity calculation and text generation

**Technologies Used:** Python, PyTorch, TorchText, NumPy

**Note:** Trained models (.pt files), vocabularies, and loss logs are generated during execution but excluded from version control.

### Assignment 3: Sequence-to-Sequence Modeling

**Key Components:**
- **RNN Encoder-Decoder:** Basic sequence-to-sequence architecture
- **Attention Mechanism:** Enhanced encoder-decoder with attention
- **Character-level Processing:** Tokenization and vocabulary building
- **Training Infrastructure:** Comprehensive checkpoint system (checkpoints excluded from repo)
- **Evaluation:** BLEU score and qualitative assessment

**Technologies Used:** Python, PyTorch, Custom tokenizers

**Note:** Model checkpoints, tokenizers, trained models, and output files are generated during execution but excluded from version control.

### Assignment 4: Transformer Analysis and Attention Head Manipulation

**Objective:** Analysis and manipulation of attention heads in pre-trained Transformers

**Key Components:**
- **Evaluation Metrics:** Implementation of ROUGE-L and BLEU scores
- **Head Importance Analysis:** Gradient-based scoring of attention heads
- **Attention Head Pruning:** Systematic removal of less important heads
- **Performance Analysis:** Impact assessment of pruning on model performance
- **Visualization:** Attention pattern analysis and importance heatmaps

**Technologies Used:** Python, Transformers (Hugging Face), PyTorch, Matplotlib, Seaborn

## Key Learning Outcomes

1. **Text Classification:** Understanding both generative (Naive Bayes) and discriminative (Logistic Regression) approaches
2. **Language Modeling:** Progression from statistical n-gram models to neural approaches (FNN, RNN)
3. **Sequence-to-Sequence Learning:** Encoder-decoder architectures with and without attention
4. **Transformer Analysis:** Understanding attention mechanisms and their importance in modern NLP
5. **Model Evaluation:** Implementation of various evaluation metrics (Accuracy, Perplexity, BLEU, ROUGE-L)
6. **Deep Learning:** PyTorch implementation of neural networks for NLP tasks

## Technical Skills Demonstrated

- **Programming:** Python, PyTorch, NumPy, Pandas
- **NLP Libraries:** NLTK, Gensim, Transformers (Hugging Face), TorchText
- **Machine Learning:** Supervised learning, neural networks, attention mechanisms
- **Data Processing:** Text preprocessing, tokenization, vocabulary building
- **Model Training:** Checkpoint management, loss tracking, hyperparameter tuning
- **Evaluation:** Multiple metrics implementation and analysis
- **Visualization:** Matplotlib, Seaborn for model analysis

## Dataset Information

**Note:** Dataset files are excluded from version control as per .gitignore policy.

- **Assignment 1:** Sentiment classification dataset (positive/negative reviews) - downloaded programmatically
- **Assignment 2:** City names dataset for character-level language modeling - provided in notebook
- **Assignment 3:** Indian name transliteration dataset (English ↔ Hindi) - downloaded during execution  
- **Assignment 4:** CNN/DailyMail and WMT datasets for transformer analysis - loaded via Hugging Face datasets

All datasets are downloaded or generated during notebook execution and are not stored in the repository.

## Evaluation and Grading

All assignments use auto-evaluation systems with specific output formats. The code follows strict guidelines for:
- Function implementation within designated code blocks
- Preservation of evaluation print statements
- Consistent file naming conventions
- Proper checkpoint and model saving

## Repository Philosophy

This repository follows a clean version control approach, focusing on:
- **Source notebooks** (.ipynb files) containing the core assignment implementations
- **Essential documentation** (README.md, introduction.md)
- **Key visualization results** (PNG files for Assignment 4 analysis)
- **Submitted code files** (selected .py exports for assignment submissions)

**Excluded from version control:**
- Large model files and training artifacts
- Generated datasets and intermediate outputs  
- System-specific files and temporary artifacts
- Auto-generated content that can be reproduced by running the notebooks

This approach keeps the repository lightweight while preserving the essential intellectual work and reproducible implementations.

## Usage Instructions

1. **Environment Setup:** Python 3.9-3.11 recommended
2. **Dependencies:** Install required packages as specified in each notebook
3. **Execution:** Run notebooks sequentially, following cell order
4. **GPU Recommendation:** Some assignments (especially 3 and 4) benefit from GPU acceleration
5. **File Organization:** Maintain the specified directory structure for proper evaluation

## Notes

- All implementations are original and follow the provided specifications
- Checkpoints are saved for long-running training processes
- Evaluation metrics and output formats are preserved for auto-grading

---

*This repository represents comprehensive coursework in Natural Language Processing, covering both traditional and modern approaches to various NLP tasks.*
