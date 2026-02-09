#!/usr/bin/env python3
"""
Tokenization Analysis and Implementation
CS5760 Natural Language Processing - Homework 1, Q5
Student: Sai Charan Reddy Chitla
University of Central Missouri

This script demonstrates different tokenization approaches and compares
naive tokenization with NLTK's word tokenizer.
"""

import re
try:
    import nltk
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not installed. Install with: pip install nltk")


class Tokenizer:
    """Custom tokenizer with various tokenization strategies."""
    
    def __init__(self):
        self.multiword_expressions = [
            "New York City",
            "by the way",
            "machine learning",
            "artificial intelligence",
            "natural language processing",
            "United States",
        ]
    
    def naive_tokenize(self, text):
        """
        Naive space-based tokenization.
        Simply splits on whitespace.
        
        Args:
            text: Input string
            
        Returns:
            List of tokens
        """
        return text.split()
    
    def manual_tokenize(self, text):
        """
        Manual tokenization handling punctuation, contractions, and clitics.
        
        Args:
            text: Input string
            
        Returns:
            List of tokens
        """
        tokens = []
        
        # Split on whitespace first
        words = text.split()
        
        for word in words:
            # Handle contractions and possessives
            # Common patterns: can't, won't, it's, we're, I'm, you're, etc.
            
            # Special cases for common contractions
            contraction_patterns = [
                (r"n't\b", ["n't"]),  # can't, won't, don't
                (r"'re\b", ["'re"]),  # we're, you're, they're
                (r"'ve\b", ["'ve"]),  # I've, we've, they've
                (r"'ll\b", ["'ll"]),  # I'll, we'll, they'll
                (r"'m\b", ["'m"]),    # I'm
                (r"'d\b", ["'d"]),    # I'd, we'd, they'd
                (r"'s\b", ["'s"]),    # it's, he's, what's
            ]
            
            # Check if word contains a contraction
            has_contraction = False
            for pattern, replacement in contraction_patterns:
                if re.search(pattern, word):
                    # Split before the contraction
                    parts = re.split(f'({pattern})', word)
                    for part in parts:
                        if part and part.strip():
                            # Remove trailing punctuation from the base word
                            if re.match(r'^[A-Za-z]+$', part):
                                tokens.append(part)
                            elif re.search(pattern, part):
                                tokens.append(part)
                            else:
                                # Handle punctuation
                                self._split_punctuation(part, tokens)
                    has_contraction = True
                    break
            
            if not has_contraction:
                # No contraction, just split punctuation
                self._split_punctuation(word, tokens)
        
        return tokens
    
    def _split_punctuation(self, word, tokens):
        """
        Helper function to split punctuation from words.
        
        Args:
            word: Word potentially containing punctuation
            tokens: List to append tokens to
        """
        # Pattern to separate leading/trailing punctuation
        # Keep internal hyphens and apostrophes (for compounds like state-of-the-art)
        pattern = r'^([^\w\s]+)?([A-Za-z0-9\-\']+)?([^\w\s]+)?$'
        match = re.match(pattern, word)
        
        if match:
            leading_punct, core, trailing_punct = match.groups()
            
            if leading_punct:
                for char in leading_punct:
                    tokens.append(char)
            
            if core:
                tokens.append(core)
            
            if trailing_punct:
                for char in trailing_punct:
                    tokens.append(char)
        elif word:
            # Fallback: just add the word as-is
            tokens.append(word)
    
    def identify_mwes(self, text):
        """
        Identify multiword expressions in text.
        
        Args:
            text: Input string
            
        Returns:
            List of identified MWEs
        """
        found_mwes = []
        for mwe in self.multiword_expressions:
            if mwe.lower() in text.lower():
                found_mwes.append(mwe)
        return found_mwes


def compare_tokenizers(text, tokenizer):
    """
    Compare different tokenization approaches.
    
    Args:
        text: Text to tokenize
        tokenizer: Tokenizer instance
    """
    print("=" * 80)
    print("TOKENIZATION COMPARISON")
    print("=" * 80)
    
    print(f"\nOriginal text:\n{text}\n")
    
    # Naive tokenization
    naive_tokens = tokenizer.naive_tokenize(text)
    print(f"1. Naive space-based tokenization ({len(naive_tokens)} tokens):")
    print(f"   {naive_tokens}\n")
    
    # Manual tokenization
    manual_tokens = tokenizer.manual_tokenize(text)
    print(f"2. Manual tokenization ({len(manual_tokens)} tokens):")
    print(f"   {manual_tokens}\n")
    
    # NLTK tokenization (if available)
    if NLTK_AVAILABLE:
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt')
        
        nltk_tokens = word_tokenize(text)
        print(f"3. NLTK word_tokenize ({len(nltk_tokens)} tokens):")
        print(f"   {nltk_tokens}\n")
        
        # Compare manual vs NLTK
        print("=" * 80)
        print("COMPARISON: Manual vs NLTK")
        print("=" * 80)
        
        differences = []
        if len(manual_tokens) != len(nltk_tokens):
            print(f"\nToken count difference: Manual={len(manual_tokens)}, NLTK={len(nltk_tokens)}")
        
        # Find differences
        for i, (m_tok, n_tok) in enumerate(zip(manual_tokens, nltk_tokens)):
            if m_tok != n_tok:
                differences.append((i, m_tok, n_tok))
        
        if differences:
            print(f"\n{len(differences)} differences found:")
            for idx, manual, nltk_tok in differences[:10]:  # Show first 10
                print(f"  Position {idx}: Manual='{manual}' vs NLTK='{nltk_tok}'")
        else:
            print("\nNo differences found! Manual tokenization matches NLTK.")
    else:
        print("3. NLTK tokenizer: Not available (install nltk to compare)")
    
    # Identify MWEs
    print("\n" + "=" * 80)
    print("MULTIWORD EXPRESSIONS (MWEs)")
    print("=" * 80)
    
    mwes = tokenizer.identify_mwes(text)
    if mwes:
        print(f"\nFound {len(mwes)} multiword expression(s):")
        for mwe in mwes:
            print(f"  - '{mwe}'")
        
        print("\nWhy these should be single tokens:")
        print("  • They represent unified semantic concepts")
        print("  • Individual words don't contribute their literal meanings")
        print("  • Treating them as units improves downstream NLP tasks")
    else:
        print("\nNo predefined MWEs found in this text.")
        print("\nCommon MWEs include:")
        for mwe in tokenizer.multiword_expressions[:3]:
            print(f"  - {mwe}")


def demonstrate_tokenization_challenges():
    """Demonstrate various tokenization challenges."""
    
    print("\n" + "=" * 80)
    print("TOKENIZATION CHALLENGES")
    print("=" * 80)
    
    challenges = [
        ("Contractions", "I can't believe it's already done!"),
        ("Possessives", "John's book is on Mary's desk."),
        ("Abbreviations", "Dr. Smith works at the U.S. embassy."),
        ("Numbers", "The price is $19.99 or €15.50."),
        ("URLs/Emails", "Visit example.com or email test@example.com"),
        ("Hyphenated", "state-of-the-art machine learning"),
    ]
    
    tokenizer = Tokenizer()
    
    for challenge_type, text in challenges:
        print(f"\n{challenge_type}:")
        print(f"  Text: {text}")
        naive = tokenizer.naive_tokenize(text)
        manual = tokenizer.manual_tokenize(text)
        print(f"  Naive:  {naive}")
        print(f"  Manual: {manual}")


def main():
    """Main function to demonstrate tokenization."""
    
    # The paragraph from Q5
    paragraph = (
        "I can't believe it's already 2026! The AI revolution has transformed "
        "everything. We're seeing unprecedented changes in technology, society, "
        "and even how we communicate."
    )
    
    # Create tokenizer
    tokenizer = Tokenizer()
    
    # Compare tokenization approaches
    compare_tokenizers(paragraph, tokenizer)
    
    # Demonstrate various challenges
    demonstrate_tokenization_challenges()
    
    # Reflection
    print("\n" + "=" * 80)
    print("REFLECTION")
    print("=" * 80)
    
    print("""
Tokenization Challenges in English:

1. HARDEST PART:
   The most challenging aspect of English tokenization is handling ambiguous 
   punctuation. Apostrophes can indicate possession (John's), contractions 
   (can't), or informal speech ('til). Periods mark sentences, abbreviations 
   (Dr., U.S.), or decimal numbers (3.14).

2. COMPARISON WITH OTHER LANGUAGES:
   Compared to languages with clear word boundaries (like spaces in English),
   English tokenization is relatively straightforward. However, languages like
   Chinese and Japanese have no spaces between words, requiring complex 
   segmentation algorithms. Languages with rich morphology like Turkish or 
   Finnish face different challenges - single words can encode information 
   requiring multiple English tokens.

3. PUNCTUATION COMPLEXITY:
   Punctuation must be carefully separated without breaking meaningful units.
   For example:
   - "example.com" should stay together (not "example . com")
   - "Ph.D." should be one token
   - "$19.99" should keep the currency symbol attached

4. MORPHOLOGY:
   English morphology adds complexity through:
   - Contractions (can't → ca + n't)
   - Possessives (John's → John + 's)
   - Compound words (machine learning - one or two tokens?)

5. MULTIWORD EXPRESSIONS:
   MWEs significantly complicate tokenization because they require looking
   beyond whitespace to identify multi-token units functioning as single
   semantic units. This often requires named entity recognition or phrase
   detection systems.

CONCLUSION:
While English tokenization seems straightforward due to space-delimited words,
handling edge cases correctly requires sophisticated rules or machine learning
models. Modern tokenizers like NLTK's word_tokenize embody years of linguistic
research to handle these complexities consistently.
""")


if __name__ == "__main__":
    main()
