# B-simple_python_script.py
print("Hello, HPC from Python!")


# Calculate Fibonacci sequence
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        print(a, end=" ")
        a, b = b, a + b
    print()


print("Fibonacci sequence up to 10:")
fibonacci(10)

# You can also do some NLP stuff

import re

text = "Hello, HPC! This is an example sentence for basic NLP tasks."

# Tokenization
tokens = re.findall(r"\b\w+\b", text)
print("Tokens:", tokens)

# Counting word frequencies
from collections import Counter

word_freq = Counter(tokens)
print("Word Frequencies:", word_freq)
