#!/usr/bin/env python3
"""
BPE (Byte Pair Encoding) Learner
CS5760 Natural Language Processing - Homework 1, Q2.2
Student: Sai Charan Reddy Chitla
University of Central Missouri

This script implements a Byte Pair Encoding algorithm to learn subword tokenization
from a toy corpus.
"""

from collections import defaultdict, Counter
import re


class BPELearner:
    """A simple BPE learner for subword tokenization."""
    
    def __init__(self):
        self.merges = []  # List of learned merge operations
        self.vocab = set()  # Current vocabulary
        
    def get_vocab(self, corpus):
        """
        Initialize vocabulary with character-level tokens.
        Add end-of-word marker '_' to each word.
        
        Args:
            corpus: List of words
            
        Returns:
            Dictionary mapping words to their character sequences
        """
        vocab = defaultdict(int)
        for word in corpus:
            # Add end-of-word marker and split into characters
            word_chars = ' '.join(list(word)) + ' _'
            vocab[word_chars] += 1
        return vocab
    
    def get_stats(self, vocab):
        """
        Count all adjacent character pairs in the vocabulary.
        
        Args:
            vocab: Dictionary of word -> frequency
            
        Returns:
            Counter of bigram -> total frequency
        """
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs
    
    def merge_vocab(self, pair, vocab):
        """
        Merge the most frequent pair in the vocabulary.
        
        Args:
            pair: Tuple of (symbol1, symbol2) to merge
            vocab: Current vocabulary
            
        Returns:
            Updated vocabulary with merged pairs
        """
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word in vocab:
            # Replace the pair with merged version
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = vocab[word]
        
        return new_vocab
    
    def learn_bpe(self, corpus, num_merges=10):
        """
        Learn BPE merges from corpus.
        
        Args:
            corpus: List of words
            num_merges: Number of merge operations to perform
        """
        vocab = self.get_vocab(corpus)
        
        print("=" * 60)
        print("BPE Learning Process")
        print("=" * 60)
        print(f"\nCorpus: {' '.join(corpus)}")
        print(f"Number of merges to perform: {num_merges}\n")
        
        # Initial vocabulary
        self.vocab = set()
        for word in vocab:
            self.vocab.update(word.split())
        
        print(f"Initial vocabulary size: {len(self.vocab)}")
        print(f"Initial vocabulary: {sorted(self.vocab)}\n")
        
        for i in range(num_merges):
            pairs = self.get_stats(vocab)
            
            if not pairs:
                print(f"\nNo more pairs to merge after {i} iterations.")
                break
            
            # Get most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            print(f"Merge {i + 1}:")
            print(f"  Most frequent pair: {best_pair} (count: {pairs[best_pair]})")
            print(f"  Merging '{best_pair[0]}' + '{best_pair[1]}' → '{best_pair[0]}{best_pair[1]}'")
            
            # Perform merge
            vocab = self.merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)
            
            # Update vocabulary
            new_token = ''.join(best_pair)
            self.vocab.add(new_token)
            
            print(f"  New token added: '{new_token}'")
            print(f"  Vocabulary size: {len(self.vocab)}")
            print()
        
        return vocab
    
    def segment_word(self, word, vocab):
        """
        Segment a word using learned BPE merges.
        
        Args:
            word: Word to segment
            vocab: Final vocabulary from training
            
        Returns:
            List of subword tokens
        """
        # Add end-of-word marker
        word_chars = ' '.join(list(word)) + ' _'
        
        # Apply merges in order
        for pair in self.merges:
            bigram = ' '.join(pair)
            replacement = ''.join(pair)
            word_chars = word_chars.replace(bigram, replacement)
        
        return word_chars.split()


def main():
    """Main function to demonstrate BPE learning."""
    
    # Toy corpus from class
    corpus = ['low'] * 5 + ['lowest'] * 2 + ['newer'] * 6 + ['wider'] * 3 + ['new'] * 2
    
    # Create BPE learner
    bpe = BPELearner()
    
    # Learn BPE with 10 merges
    final_vocab = bpe.learn_bpe(corpus, num_merges=10)
    
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    
    print(f"\nLearned {len(bpe.merges)} merges:")
    for i, (a, b) in enumerate(bpe.merges, 1):
        print(f"  {i}. '{a}' + '{b}' → '{a}{b}'")
    
    print(f"\nFinal vocabulary size: {len(bpe.vocab)}")
    print(f"Final vocabulary: {sorted(bpe.vocab)}")
    
    # Test segmentation
    print("\n" + "=" * 60)
    print("Word Segmentation Examples")
    print("=" * 60)
    
    test_words = ['new', 'newer', 'lowest', 'widest', 'newestest']
    
    for word in test_words:
        segments = bpe.segment_word(word, final_vocab)
        print(f"\n'{word}' → {segments}")
    
    # Analysis
    print("\n" + "=" * 60)
    print("Analysis: How Subwords Solve the OOV Problem")
    print("=" * 60)
    
    print("""
The subword tokenization approach solves the Out-of-Vocabulary (OOV) problem
by breaking unknown words into known subword units. Instead of treating each 
word as an atomic unit, BPE learns frequently occurring character sequences 
and treats them as tokens.

For example, even though 'newestest' never appeared in our training corpus,
it can still be segmented using the learned subword tokens. This allows the
model to handle rare or invented words by composing them from familiar pieces.

Importantly, many subword tokens align with meaningful morphemes. In our
example, 'er_' consistently appears as a suffix marker for comparative 
adjectives (newer, wider) and can also mark agent nouns in English. This
demonstrates how BPE can discover linguistically meaningful units without
explicit morphological knowledge.

The '_' marker at the end of words is crucial - it distinguishes word-final
tokens from word-internal ones, allowing the model to learn that 'er_' at
the end of a word has different properties than 'er' in the middle.
""")


if __name__ == "__main__":
    main()
