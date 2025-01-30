#!/usr/bin/env python

import requests
import re
import multiprocessing
from collections import Counter
import matplotlib.pyplot as plt

def fetch_text(url):
    """Downloads text from the specified URL."""
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def tokenize(text, min_characters=3):
    """Breaks text into words and cleans it of unwanted characters."""
    pattern = fr'\b\w{{{min_characters},}}\b'
    words = re.findall(pattern, text.lower())  # Minimum {min_characters} characters to filter. Default: min_characters=3
    return words

def mapper(chunk):
    """Map function: creates a pair (word, 1) for each word."""
    return Counter(chunk)

def reducer(counter1, counter2):
    """Reduce function: combines two counters."""
    counter1.update(counter2)
    return counter1

def map_reduce(text, min_characters, num_workers=4):
    """Applies the MapReduce paradigm to calculate word frequency."""
    words = tokenize(text, min_characters)
    chunk_size = len(words) // num_workers
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
    
    with multiprocessing.Pool(num_workers) as pool:
        mapped = pool.map(mapper, chunks)
    
    result = Counter()
    for partial_result in mapped:
        result = reducer(result, partial_result)
    
    return result

def visualize_top_words(word_freq, min_characters, top_n=10):
    """Visualizes the top-N most frequently used words."""
    top_words = word_freq.most_common(top_n)
    words, counts = zip(*top_words)
    
    plt.figure(figsize=(10, 6))
    plt.barh(words, counts, color='skyblue')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.title(f'Top {top_n} Most Frequently Used Words. Minimum characters={min_characters}')
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == "__main__":
    url = "https://www.gutenberg.org/cache/epub/5046/pg5046.txt"  # Example: "State of the Union Addresses of Ronald Reagan"
    min_characters=4
    num_workers=5
    top_n=10

    print("[*] Loading text...")
    text = fetch_text(url)
    
    print("[*] Performing MapReduce...")
    word_freq = map_reduce(text, min_characters, num_workers)
    
    print("[*] Visualization of results...")
    visualize_top_words(word_freq, min_characters, top_n)
    print("[*] Done.")
