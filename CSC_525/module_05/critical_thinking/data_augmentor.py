import random
import re
import nltk
from nltk.corpus import wordnet
import argparse
# ------------------------------------------------------------------------------------------------------------
# Data Augmentation Script
#
# This script performs Easy Data Augmentation (EDA) techniques on text data, including synonym replacement,
# random deletion, random swap, and random insertion. It is designed to enhance the diversity of text data
# for machine learning tasks.
#
# Usage:
#   $ python data_augmentor.py <input_file> <output_file> --num_aug <number_of_augmentations>
# ------------------------------------------------------------------------------------------------------------
def get_synonyms(word):
    """Get a list of synonyms for a given word using WordNet.

    Args:
        word (str): The word to find synonyms for.

    Returns:
        list: A list of synonyms for the word.
    """
    synonyms = set()
    for syn in wordnet.synsets(word): # Iterate over all synsets of the word.
        for l in syn.lemmas(): # Iterate over all lemmas in the synset.
            synonym = l.name().replace("_", " ").lower() # Replace underscores and convert to lowercase.
            if synonym != word: # Avoid adding the word itself as a synonym.
                synonyms.add(synonym)
    return list(synonyms)

def synonym_replacement(words, n):
    """Replace `n` words in the list with their synonyms.

    Args:
        words (list): List of words in the sentence.
        n (int): Number of words to replace.

    Returns:
        list: List of words with synonyms replaced.
    """
    new_words = words.copy() # Create a copy of the original word list.
    # Filter words with synonyms.
    filtered_words = []
    for word in words:
        if get_synonyms(word):
            filtered_words.append(word)
    random_word_list = list(set(filtered_words)) # Shuffle the list of words.
    random.shuffle(random_word_list) # Shuffle the list of words.
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word) # Get synonyms for the word.
        if synonyms:
            synonym = random.choice(synonyms) # Randomly select a synonym.
            # Replace the word with its synonym.
            new_words = []
            for word in new_words:
                if word == random_word:
                    new_words.append(synonym)
                else:
                    new_words.append(word)
            num_replaced += 1
        if num_replaced >= n: # Stop if the required number of replacements is reached.
            break
    return new_words

def random_deletion(words, p):
    """Randomly delete words from the list with probability `p`.

    Args:
        words (list): List of words in the sentence.
        p (float): Probability of deleting each word.

    Returns:
        list: List of words after random deletion.
    """
    # Avoid deleting the only word in the sentence.
    if len(words) == 1:  
        return words
    # Retain the word with a probability greater than `p`.
    retained_words = []
    for word in words:
        if random.random() > p:  
            retained_words.append(word)
    return retained_words

def random_swap(words, n):
    """Randomly swap two words in the list `n` times.

    Args:
        words (list): List of words in the sentence.
        n (int): Number of swaps to perform.

    Returns:
        list: List of words after random swaps.
    """
    new_words = words.copy() # Create a copy of the original word list.
    for _ in range(n):
        idx1 = random.randint(0, len(new_words)-1) # Randomly select the first index.
        idx2 = random.randint(0, len(new_words)-1) # Randomly select the second index.
        # Swap the words at the selected indices.
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words

def random_insertion(words, n):
    """Randomly insert `n` synonyms into the list.

    Args:
        words (list): List of words in the sentence.
        n (int): Number of words to insert.

    Returns:
        list: List of words after random insertions.
    """
    new_words = words.copy() # Create a copy of the original word list.
    for _ in range(n):
        add_word(new_words) # Insert a random synonym.
    return new_words

def add_word(new_words):
    """Helper function to insert a random synonym into the list.

    Args:
        new_words (list): List of words in the sentence.
    """
    synonyms = []
    counter = 0
    while len(synonyms) < 1 and counter < 10: # Retry up to 10 times to find a synonym.
        # Select a random word.
        random_index = random.randint(0, len(new_words)-1)
        random_word = new_words[random_index]
        synonyms = get_synonyms(random_word) # Get synonyms for the word.
        counter += 1
    if synonyms:
        random_synonym = random.choice(synonyms) # Randomly select a synonym.
        random_idx = random.randint(0, len(new_words)-1) # Select a random index.
        new_words.insert(random_idx, random_synonym) # Insert the synonym at the selected index.

def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=42):
    """Main Augmentation Function. Performs Easy Data Augmentation (EDA) on a 
    given sentence.

    Args:
        sentence (str): The input sentence to augment.
        alpha_sr (float): Proportion of words to replace with synonyms.
        alpha_ri (float): Proportion of words to randomly insert synonyms.
        alpha_rs (float): Proportion of words to randomly swap.
        p_rd (float): Probability of deleting each word.
        num_aug (int): Number of augmented sentences to generate.

    Returns:
        list: List of augmented sentences.
    """
    # Remove punctuation from the sentence.
    sentence = re.sub(r"[^\w\s]", "", sentence)
    words = nltk.word_tokenize(sentence) # Tokenize the sentence into words.
    num_words = len(words)
    augmented_sentences = []
    # Calculate the number of words to modify for each augmentation.
    n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))
    # Apply different augmentation techniques.
    augmented_sentences.append(" ".join(synonym_replacement(words, n_sr)))
    augmented_sentences.append(" ".join(random_insertion(words, n_ri)))
    augmented_sentences.append(" ".join(random_swap(words, n_rs)))
    augmented_sentences.append(" ".join(random_deletion(words, p_rd)))
    return augmented_sentences[:num_aug]

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs="?", default="input.txt", help="Input text file with one sentence per line.")
    parser.add_argument("output_file", nargs="?", default="output.txt", help="Output file for augmented data.")
    parser.add_argument("--num_aug", type=int, default=42, help="Number of augmentations per sentence.")
    args = parser.parse_args()
    # Try to read to the provided input file.
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.")
        return
    # Write augmented sentences to output file.
    with open(args.output_file, "w", encoding="utf-8") as out:
        for line in lines:
            line = line.strip()
            if line: # Skip empty lines.
                out.write(line + "\n") # Write the original sentence.
                for aug in eda(line, num_aug=args.num_aug): # Write augmented sentences.
                    out.write(aug + "\n")
    print(f"Augmented data saved to {args.output_file}!\nAll done.")

# Big red activation button.
if __name__ == "__main__":
    print("Running data augmentation...")
    main()
