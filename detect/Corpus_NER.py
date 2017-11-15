'''

Clean Corpus.csv. Extarct Named Entities from `Texto` column, 
add them as a string to a new column `Names`.
 
'''

import pandas as pd
import nltk, re
from nltk import word_tokenize, pos_tag, ne_chunk

corpus_file = "../data/Corpus.csv" 
corpus_ner_file = "../data/Corpus_NER.csv"

# Load corpus
df_raw = pd.read_csv(corpus_file)
print(df_raw.info())
print("*** Column names: ", df_raw.columns)

# Remove no-string and empty values in 'Texto' 
df_raw = df_raw.drop('Unnamed: 8', axis=1)
df = df_raw.drop('Unnamed: 9', axis=1)
df = df[df['Texto'].apply(lambda x: type(x) == str and len(x) > 0)]

print('\n*** After cleanup ')
print(df.info(), '\n')
print(df.head())

# Unique values in `Nivel` columns
print('\n*** Nevel_1 values: ')
print(df['Nivel_1'].unique())

print('\n*** Nevel_2 values: ')
print(df['Nivel_2'].unique())

print('\n*** Nevel_3 values: ')
print(df['Nivel_3'].unique())

def parts_of_speech(corpus):
    "Returns named entity chunks in a given text"
    sentences = nltk.sent_tokenize(corpus)
    tokenized = [nltk.word_tokenize(sentence) for sentence in sentences]
    pos_tags  = [nltk.pos_tag(sentence) for sentence in tokenized]
    chunked_sents = nltk.ne_chunk_sents(pos_tags, binary=True)
    return chunked_sents

def find_entities(chunks):
    "Given list of tagged parts of speech, returns unique named entities"

    def traverse(tree):
        "Recursively traverses an nltk.tree.Tree to find named entities"
          
        entity_names = []
    
        if hasattr(tree, 'label') and tree.label:
            if tree.label() == 'NE':
                entity_names.append(' '.join([child[0] for child in tree]))
            else:
                for child in tree:
                    entity_names.extend(traverse(child))
    
        return entity_names
    
    named_entities = []
    
    for chunk in chunks:
        
        entities = sorted(list(set([word for tree in chunk
                            for word in traverse(tree)])))
        for e in entities:
            if e not in named_entities:
                named_entities.append(e)
    return named_entities

def names(text, sep):
    "Return a string with all Named Entities found in `text` separated by `sep` "
    entity_chunks  = parts_of_speech(text)
    named_entities = find_entities(entity_chunks)
    names_string = sep.join(named_entities)
    return names_string

# Extract named entities and add them as a string to a dataframe in a separate column
sep = '|'
df['Names'] = df['Texto'].apply(lambda x: names(x, sep))

print('\n*** New table')
print(df.head())

# Save new table to a `csv` file 
df.to_csv(corpus_ner_file )

