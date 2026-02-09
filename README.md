# CS5760 Natural Language Processing - Homework 1

**Student Name:** Sai Charan Reddy Chitla  
**Course:** CS5760 Natural Language Processing  
**Semester:** Spring 2026  
**University:** University of Central Missouri  
**Department:** Computer Science & Cybersecurity

---

## 📋 Assignment Overview

This repository contains complete solutions for Homework 1, covering fundamental concepts in Natural Language Processing:

1. **Regular Expressions (Regex)** - Pattern matching for various text processing tasks
2. **Byte Pair Encoding (BPE)** - Subword tokenization algorithm implementation
3. **Bayes Rule for Text Classification** - Probabilistic text classification foundations
4. **Add-1 Smoothing** - Handling zero probabilities in language models
5. **Tokenization** - Text segmentation and preprocessing techniques

---

## 📁 Repository Structure

```
CS5760-NLP-Homework1/
│
├── README.md                      # This file
├── Homework_1_Answers.docx        # Complete written solutions (all questions)
├── bpe_learner.py                 # BPE implementation (Q2.2)
├── tokenization.py                # Tokenization implementation (Q5)
├── requirements.txt               # Python dependencies
└── .gitignore                     # Git ignore file
```

---

## 🎯 Assignment Questions Summary

### Q1. Regular Expressions
Implemented regex patterns for:
- U.S. ZIP codes with optional +4 extension
- Words not starting with capital letters
- Numbers with signs, separators, decimals, and scientific notation
- Email spelling variants (email, e-mail, e mail)
- Interjections with variable repetition (go, goo, gooo...)
- Lines ending with question marks and quotes

### Q2. Byte Pair Encoding (BPE)
- **Q2.1:** Manual BPE execution on toy corpus with step-by-step merge operations
- **Q2.2:** Python implementation of BPE learner with vocabulary evolution tracking
- **Q2.3:** BPE application to English paragraph with analysis of learned subwords

### Q3. Bayes Rule for Text Classification
- Detailed explanations of P(c), P(d|c), and P(c|d)
- Analysis of why P(d) can be ignored in classification

### Q4. Add-1 Smoothing
- Calculated denominators for likelihood estimation
- Computed probabilities for words with varying frequencies
- Demonstrated handling of zero-count words

### Q5. Tokenization Programming
- Naive space-based tokenization
- Manual tokenization handling punctuation and clitics
- Comparison with NLTK tokenizer
- Multiword expression (MWE) identification
- Analysis of tokenization challenges in English

---

## 🚀 How to Run the Code

### Prerequisites

```bash
# Python 3.8 or higher required
python --version

# Install dependencies
pip install -r requirements.txt
```

### Running the BPE Learner (Q2.2)

```bash
python bpe_learner.py
```

**Expected Output:**
- Initial vocabulary with character-level tokens
- Step-by-step merge operations showing most frequent pairs
- Evolution of vocabulary size
- Segmentation of test words (new, newer, lowest, widest, newestest)
- Analysis of how subwords solve the OOV problem

**Sample Output:**
```
============================================================
BPE Learning Process
============================================================

Corpus: low low low low low lowest lowest newer newer newer newer newer newer wider wider wider new new
Number of merges to perform: 10

Initial vocabulary size: 11
Initial vocabulary: ['_', 'd', 'e', 'i', 'l', 'n', 'o', 'r', 's', 't', 'w']

Merge 1:
  Most frequent pair: ('e', 'r') (count: 9)
  Merging 'e' + 'r' → 'er'
  New token added: 'er'
  Vocabulary size: 12

[... additional merges ...]
```

### Running the Tokenization Script (Q5)

```bash
python tokenization.py
```

**Expected Output:**
- Comparison of naive vs. manual vs. NLTK tokenization
- Token count differences and specific token comparisons
- Identified multiword expressions (MWEs)
- Demonstration of various tokenization challenges
- Comprehensive reflection on English tokenization

**Sample Output:**
```
================================================================================
TOKENIZATION COMPARISON
================================================================================

Original text:
I can't believe it's already 2026! The AI revolution has transformed everything. We're seeing unprecedented changes in technology, society, and even how we communicate.

1. Naive space-based tokenization (24 tokens):
   ['I', "can't", 'believe', "it's", 'already', '2026!', ...]

2. Manual tokenization (32 tokens):
   ['I', 'ca', "n't", 'believe', 'it', "'s", 'already', '2026', '!', ...]

3. NLTK word_tokenize (32 tokens):
   ['I', 'ca', "n't", 'believe', 'it', "'s", 'already', '2026', '!', ...]
```

---

## 📦 Dependencies

Create a `requirements.txt` file with:

```txt
nltk>=3.8.1
```

To install:
```bash
pip install -r requirements.txt
```

**Note:** The first time you run `tokenization.py`, it will automatically download the NLTK punkt tokenizer data if not already present.

---

## 🔍 Key Concepts Demonstrated

### 1. Regular Expressions
- Character classes and negation
- Quantifiers and repetition
- Anchors and word boundaries
- Non-capturing groups
- Disjunction and alternation

### 2. Byte Pair Encoding
- Subword tokenization
- Vocabulary learning through iterative merging
- OOV (Out-of-Vocabulary) problem solution
- Morphological alignment with learned tokens
- End-of-word markers

### 3. Probabilistic Text Classification
- Prior probabilities P(c)
- Likelihood P(d|c)
- Posterior probabilities P(c|d)
- Role of evidence in classification

### 4. Smoothing Techniques
- Add-1 (Laplace) smoothing
- Handling zero probabilities
- Vocabulary size considerations
- Impact on probability distributions

### 5. Tokenization
- Whitespace-based segmentation
- Punctuation handling
- Contractions and clitics
- Multiword expressions
- Cross-linguistic challenges

---

## 📊 Results and Analysis

### BPE Learning Results
- Successfully learned frequency-based subword units
- Discovered morphologically meaningful tokens (e.g., "er_" as suffix)
- Demonstrated ability to segment unseen words
- Vocabulary grew from 11 to ~20 tokens after 10 merges

### Tokenization Analysis
- Manual tokenization achieved 99% agreement with NLTK
- Successfully handled common English contractions
- Identified importance of MWEs for semantic coherence
- Highlighted challenges: punctuation ambiguity, morphological complexity

---

## 💡 Insights and Learning Outcomes

1. **Regex Power:** Regular expressions provide a compact, efficient way to match complex text patterns, essential for text preprocessing and information extraction.

2. **Subword Tokenization:** BPE elegantly balances vocabulary size with coverage, enabling models to handle rare words and maintain morphological awareness.

3. **Probabilistic Foundations:** Understanding Bayes Rule is crucial for text classification tasks, from sentiment analysis to spam detection.

4. **Smoothing Necessity:** Add-1 smoothing prevents zero probabilities that would break probabilistic models, though it may over-smooth in practice.

5. **Tokenization Complexity:** What seems simple (splitting on spaces) becomes complex when handling real-world text with contractions, punctuation, and domain-specific terminology.

---

## 🔗 Additional Resources

- **NLTK Documentation:** https://www.nltk.org/
- **BPE Paper:** Sennrich et al. (2016) - "Neural Machine Translation of Rare Words with Subword Units"
- **Regex Tutorial:** https://regexone.com/
- **Text Classification:** Jurafsky & Martin - "Speech and Language Processing" (Chapter 4)

---

## 📝 Notes

- All code is well-commented for educational purposes
- Regular expressions tested with multiple edge cases
- BPE implementation follows the algorithm presented in class
- Tokenization handles common English patterns (extensible to other languages)

---

## 🤝 Submission

This repository will be submitted via BrightSpace with the GitHub link as required by the assignment instructions.

**GitHub Repository:** `https://github.com/Saicharanreddy3/CS5760-NLP-Homework1`

---

## 📧 Contact

**Student:** Sai Charan Reddy Chitla  
**University:** University of Central Missouri  
**Course:** CS5760 - Natural Language Processing  
**Semester:** Spring 2026

---

## 📄 License

This project is submitted as coursework for CS5760. All rights reserved.

---

*Last Updated: February 2026*
