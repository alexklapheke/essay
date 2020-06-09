#!/usr/bin/env python3

# Evaluates grade 10 level expository essays on a scale from 0 (worst)
# to 3 (best).

import sys

freq_col = "freq_per_100000"
pad_shape = 1000


def freq_score(doc):
    """Calculates an arbitrarily-defined "frequency metric"
    that tries to capture how many infrequent words are used
    in the text."""
    # Read in American English word frequency list
    freqs = pd.read_csv("../data/anc_frequency_list.csv")

    # Merge homonyms, as they won't affect the analysis much
    freqs = pd.DataFrame(freqs.groupby("lemma")[freq_col].sum())

    # Add rank
    freqs["rank"] = freqs[freq_col].rank(method="min",
                                         ascending=False).astype(int)

    score = 0
    for word in doc:
        lemma = word.lemma_
        try:
            score += (freqs.loc[lemma, "rank"])
        except KeyError:
            pass
    return score


def is_word(w):
    return (not w.is_space) and (not w.is_punct)


def preprocess(essay):
    with open("../data/english-linking-words.txt") as file:
        linking_words = file.read().split("\n")[:-1] # drop final newline

    doc = nlp(essay)

    metrics = {}

    # Token count
    metrics["tokens"] = len(list(filter(is_word, doc)))

    # Type count (may be inflated by misspellings)
    metrics["types"] = len(set([word.lemma_ for word in doc if is_word(word)]))

    # Mean sentence length
    metrics["sent_len"] = np.mean([len(list(filter(is_word, s)))
                                   for s in doc.sents])

    # Mean word length
    metrics["word_len"] = np.mean(list(map(len, filter(is_word, doc))))

    # Word frequency measure
    metrics["freq"] = freq_score(doc)

    # Semicolons per token
    metrics["semicolons"] = len([token for token in doc
                                 if token.text == ";"]) / metrics["tokens"]

    # Linking words per token
    metrics["link_words"] = sum([len(re.findall("\\b{}\\b".format(link), essay))
                                 for link in linking_words]) / metrics["tokens"]

    # Number of prepositional phrases per token
    metrics["pps"] = len([tok.pos_ == "ADP" for tok in doc]) / metrics["tokens"]

    # Depth of longest branch in dependency tree
    metrics["max_depth"] = max([len(list(tok.children)) for tok in doc])

    return metrics


def run_model(essay, metrics, model, modeltype="nn"):
    meta_cols = [
        "tokens",
        "types",
        "sent_len",
        "word_len",
        "freq",
        "semicolons",
        "link_words",
        "pps",
        "max_depth",
    ]

    X_meta = pd.DataFrame(metrics, index=[0])[meta_cols]
    X_meta_sc = ss.transform(X_meta)
    X_meta_pca = pca.transform(X_meta_sc)

    if modeltype == "nn":
        word2idx = {u: i for i, u in enumerate(vocab)}

        X_vector = [word2idx[token.text] for token in nlp.tokenizer(essay)]
        X_vector = pad_sequences([X_vector], maxlen=pad_shape)

        return model.predict([X_vector, X_meta_pca]).argmax()

    elif modeltype == "svm":
        return int(model.predict(X_meta_sc)[0])


if __name__ == '__main__':
    try:
        essayfile = sys.argv[1]
        if essayfile == '-':
            essay = sys.stdin.read().strip()
        else:
            with open(essayfile, "r") as infile:
                essay = infile.read().strip()
    except IndexError:
        essay = input("Type your essay:\n")

    import pandas as pd
    import numpy as np
    from joblib import load
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import spacy
    import re

    print("Loading NLP data...")
    nlp = spacy.load("en")

    print("Loading model...")
    vocab = load("vocab.bin")
    ss = load("scaler.bin")
    pca = load("pca.bin")

    # model = load_model("model.keras")
    model = load("model.svm")

    print("Preprocessing essay...")
    metrics = preprocess(essay)

    print("Running model...")
    print("Score:", run_model(essay, metrics, model, modeltype="svm"))
