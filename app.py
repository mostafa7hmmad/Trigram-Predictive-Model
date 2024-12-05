from collections import Counter, defaultdict
import math
file_path = r"C:\Users\mostafa\AI&ML\assiment-nlp\AI-work.txt"

def collect_and_tokenize_corpus(file_path):
    # Read file contents
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()  # Convert to lowercase for consistency
    
    # Simple tokenizer (splitting by spaces)
    tokens = text.split()
    
    # Create trigrams
    trigram_model = [(tokens[i], tokens[i+1], tokens[i+2]) for i in range(len(tokens) - 2)]
    
    return tokens, trigram_model
def build_trigram_model(trigram_model, tokens, alpha=1):
    trigram_counts = Counter(trigram_model)
    total_trigrams = sum(trigram_counts.values())
    vocabulary_size = len(set(tokens))
    
    # Calculate probabilities with Laplace Smoothing
    smoothed_probs = defaultdict(lambda: alpha / (total_trigrams + alpha * vocabulary_size))
    
    for trigram, count in trigram_counts.items():
        smoothed_probs[trigram] = (count + alpha) / (total_trigrams + alpha * vocabulary_size)
    
    return trigram_counts, smoothed_probs
def autocomplete(input_text, smoothed_probs):
    input_tokens = input_text.lower().split()
    if len(input_tokens) < 2:
        return ["Please type at least two words."]
    
    last_bigram = tuple(input_tokens[-2:])
    suggestions = {
        trigram[-1]: prob for trigram, prob in smoothed_probs.items() if trigram[:2] == last_bigram
    }
    
    if not suggestions:
        return ["No suggestions available."]
    
    sorted_suggestions = sorted(suggestions.items(), key=lambda x: -x[1])[:5]
    return [word for word, _ in sorted_suggestions]
def calculate_perplexity(smoothed_probs, test_trigrams):
    N = len(test_trigrams)
    log_prob_sum = 0
    for trigram in test_trigrams:
        prob = smoothed_probs[trigram]
        log_prob_sum += math.log(prob, 2)  # Log base 2 for perplexity calculation
    perplexity = 2 ** (-log_prob_sum / N)
    return perplexity
from collections import Counter, defaultdict

tokens, trigram_model = collect_and_tokenize_corpus(file_path)
trigram_counts, smoothed_probs = build_trigram_model(trigram_model, tokens)
import streamlit as st

st.title("Trigram Model Autocomplete and Perplexity Calculator")

# Display basic model information
st.write(f"Total Words in Corpus: {len(tokens)}")
st.write(f"Total Trigrams in Corpus: {len(trigram_model)}")
st.write(f"Vocabulary Size: {len(set(tokens))}")

# Autocomplete Section
st.header("Autocomplete Suggestions")
user_input = st.text_input("Type your text here:")

if user_input:
    suggestions = autocomplete(user_input, smoothed_probs)
    st.write("Suggestions:")
    for suggestion in suggestions:
        st.write(suggestion)
