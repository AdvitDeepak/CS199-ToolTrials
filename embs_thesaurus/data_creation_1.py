import random
import nltk
from nltk.corpus import wordnet

nltk.download('words')
nltk.download('wordnet')


def generate_synonym_antonym_lines(num_lines):
    with open('usage_dataset2.txt', 'w') as file:
        for _ in range(num_lines):
            word = generate_random_word()
            synonym = get_synonym(word)
            antonym = get_antonym(word)

            if synonym != word: 
                line1 = f"{word} <FUNC1_STT> {synonym} \n"
                file.write(line1)

            if antonym != word: 
                line2 = f"{word} <FUNC2_STT> {antonym} \n"
                file.write(line2)


def generate_random_word():
    word_list = nltk.corpus.words.words()
    word = random.choice(word_list)
    return word


def get_synonym(word):
    synsets = wordnet.synsets(word)
    synonyms = set()

    for syn in synsets:
        for lemma in syn.lemmas():
            if lemma.name() != word:
                synonyms.add(lemma.name())

    if synonyms:
        return random.choice(list(synonyms))
    else:
        return word

def get_antonym(word):
    synsets = wordnet.synsets(word)
    antonyms = set()

    for syn in synsets:
        for lemma in syn.lemmas():
            if lemma.name() != word and lemma.antonyms():
                antonyms.add(lemma.antonyms()[0].name())

    if antonyms:
        return random.choice(list(antonyms))
    else:
        return word

# Example usage
num_lines = 10000
generate_synonym_antonym_lines(num_lines)
print(f"{num_lines} lines have been generated and saved to 'usage_dataset.txt'.")

