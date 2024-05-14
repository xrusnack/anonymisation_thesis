import re, pickle
import pandas as pd
from pandas import DataFrame
from typing import List, Dict, Any, Tuple

import numpy as np
np.int = np.int32
np.float = np.float64
np.bool = np.bool_

import nltk as nltk_lib

import tokenizations
from textspan import get_original_spans
from clinitokenizer.tokenize import clini_tokenize

from deid_utils import add_basic_features, add_contextual_features, assign_sentence_id, satisfies_regex_rule, crf_feature_format, group_words_into_sentences, mask_text_crf


# Specify the path where the de-identified text is to be saved.
PATH = 'deid_text.txt'

# Put the text you want to de-identify here.
TEXT = """
[ukazka]
Makroskopický nález: Zpracováno kompletně. PŘÍJEM: 20.12.2024 v 12:00 hod PROVEDENO: 21.12.2024 v 12:00 hod PROVEDL: MUDr. Janko Hruška.
"""


def deidentify(input: str) -> str:
    """
    This function takes a text for de-identification as an input. It loads an 
    already pretrained crf model and predicts labels for each word in text. 
    As a result, words for anonymization predictied by the model are masked
    with letter 'X'. It returns a de-identified version of the text which is
    exported as a 'deid_text.txt' file.
    """
    try:
        nltk_lib.data.find("tokenizers/punkt")
    except LookupError:
        nltk_lib.download('punkt')

    if input.strip() == "":
        return input

    text = input.replace('"', " ").replace("'", " ")
    tokens: List[str] = nltk_lib.tokenize.word_tokenize(text, language='czech', preserve_line=False)
    offsets = [item for sublist in get_original_spans(tokens, text) for item in sublist]
    tokens_with_offsets: List[Dict[str, Any]] = [{"word": word, 
                                                  "start": offset[0], 
                                                  "end": offset[1]} for word, offset in zip(tokens, offsets)]
    data = pd.DataFrame(tokens_with_offsets, columns=["sentence", "word", "start", "end", "regex_rule"])
    satisfies_regex_rule(text, data)
        
    sentences = clini_tokenize(text)
    assign_sentence_id(data, sentences)
    data["sentence"] = data["sentence"].astype(int)

    with open('./models/crf_model.pkl', 'rb') as file:
        crf_model = pickle.load(file)

    data = group_words_into_sentences(data)
    data_crf = add_basic_features(data, True).drop(["sentence", "start", "end"], axis=1)
    for n in range(1, 3):
        data_crf = add_contextual_features(data_crf, True, n)

    predicted = crf_model.predict(crf_feature_format(data_crf))
    data["predicted"] = predicted

    return mask_text_crf(input, data)
                

def main():    
    deidentified: str = deidentify(TEXT)
    
    with open(PATH, 'w') as file:
        file.write(deidentified) 


if __name__ == '__main__':
    main()