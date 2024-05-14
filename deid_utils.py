import re
import pandas as pd
from typing import List, Dict, Any, Tuple
from pandas import DataFrame, Series
from corpy.morphodita import Tagger


DOCTOR_PATTERN = r'''
    \b                                         # match only if it appears at the start of a word,
    (?:                                        # a doctor has 1 or more titles,
       [Dd]r\.?(?:\s)?|                     
       MUD[Rr]\.?(?:\s)?|
       MDD[Rr]\.?(?:\s)?|
       [Pp]harm[Dd][Rr]\.?(?:\s)?|
       doc\.?(?:\s)?|
       prim\.?(?:\s)?|
       Mgr\.?(?:\s)?
    )+
    (?:
       (?!\n)\s                                # a (trailing) space character
    )*
    [A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]*                      # a first name (or a surname) - allow for the first letter to be in lower case,
    [a-záčďéěíňóřšťúůýž]+                      
    (?:                                        # none or 1 surname (must have the first capital letter),
       (?:                                     
       (?!\n)\s                                # a (trailing) space character
       )+
       [A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]
       [a-záčďéěíňóřšťúůýž]+
    )?
    ,?
    (?:                                        # other possible titles
       (?:(?!\n)\s)+
          [Pp][Hh]\.?[Dd]\.?|
       (?:(?!\n)\s)+
          [Cc][Ss]c\.?|
       (?:(?!\n)\s)+
          [Dd][Rr][Ss]c\.?|
       (?:(?!\n)\s)+
          doc\.?
    )?
    '''

DATE_PATTERN = r'''
    \b                                       # match only if it appears at the start of a word,
    (?:
       (?:[0123]?[0-9])                      # DD.MM.YY or DD/MM/YY or DD.MM.YYYY or DD/MM/YYYY
       (?:\.|/)
       (?:[01]?[0-9])
       (?:\.|/)
       (?:[12]\d{3}|\d{2})
       |  
       (?:[0123]?[0-9])\.(?:[01]?[0-9])\.    # DD.MM.
    )
    '''

TIME_PATTERN = r'''
    \b                         # match only if it appears at the start of a word,
    (?:                         
       (?:[012]?[0-9])         # matches e.g., "HH:MM h" or "HH:MM:SS  hod." or "HH:MM"
       \:                   
       [0-5][0-9]        
       (?:
          \:[0-5][0-9]
       )?
       (
        (?:\s+)?               # trailing spaces
        (?:hod|h)?
        \.?
       )?
       |                  
       (?:[012]?[0-9])         # "HH,MM h" or "HH,MM hod."
       (?:,|\.)
       [0-5][0-9]
       (?:\s+)?                # trailing spaces
       h(?:od)?\.?
    )
    '''

PHONE_PATTERN = r'''
    \b                         # \b ... \b match only if it's a stand-alone word
    (?:
       tel\.?\s*|
       č\.?\s*
    )*
    \d{3}\s*\d{3}\s*\d{3}
    \b
'''


REGEX_PATTERNS = {
    "doktor": DOCTOR_PATTERN,
    "datum" : DATE_PATTERN,
    "čas" : TIME_PATTERN,
    "telefon" : PHONE_PATTERN
}


def mask_text_crf(text: str, data: DataFrame) -> str:
    """
    This function takes a text and a corresponding dataframe contatning offsets
    for words that are to be masked. As a result, the specified tokens are masked
    with letter 'X'.
    """
    deidentified_text = list(text)

    for index, row in data.iterrows():
        for i, elem in enumerate(row['predicted']):
            if elem == 'A':
                start = row['start'][i]
                end = row['end'][i]
                deidentified_text[start:end] = ['X'] * (end - start)
    return ''.join(deidentified_text)
    

def group_words_into_sentences(data: DataFrame) -> DataFrame:
    """
    This function groups the data provided on the input by "id" and "sentence" 
    columns, and then aggregates the remaining columns into lists for each group.
    """
    groups: List[str] = []
    
    if "id" in data.columns:
        groups.append("id")
    if "sentence" in data.columns:
        groups.append("sentence")

    assert len(groups) > 0
    
    return data.groupby(groups).agg({
            col: lambda x: list(x) for col in data.columns if col not in groups
        }).reset_index()


def unroll_grouped_data(grouped_data: DataFrame) -> DataFrame:
    """
    This is an inverse funtion to the function 'group_words_into_sentences(DataFrame)'
    """
    if isinstance(grouped_data, Series):
        grouped_data = grouped_data.to_frame()
    columns = [col for col in grouped_data.columns if col not in ["id", "sentence"]]
    unrolled_data = grouped_data.explode(columns)
    return unrolled_data.reset_index(drop=True)


def extract_entities(text: str) -> List[Dict[str, Any]]:
    """
    This function applies regex rules defined in the predefined dictionary 
    and uses them to extract entities. 
    It returns the list of extracted entities from the text, in the 
    following annotation format with keys: {start, end, text, labels}.
    """
    entities: List[Dict[str, Any]] = []
    
    for label, pattern in REGEX_PATTERNS.items():
        pattern = re.compile(pattern, re.VERBOSE)
        matches = re.finditer(pattern, text)
        
        for match in matches:
            start = match.start()
            end = match.end()
            entities.append({"start": start, 
                             "end": end, 
                             "text": match.group(), 
                             "labels": label})
    return entities


def apply_regex(text: str, regex_patterns: Dict[str, str]) -> str:
    """
    This function takes text (a string) that is to be masked and a dictionary
    of entities and their respective regex rules as and input. 
    It returns the same text with masked entities extracted using the regex dictionary.
    """
    masked = text

    for label, pattern in regex_patterns.items():
        pattern = re.compile(pattern, re.VERBOSE)
        masked = re.sub(pattern, "[" + label.upper() + "]", masked)
    return masked   


def REGEX_postprocessing(X_sentences: DataFrame, y_predictions: List[List[str]]) -> List[List[str]]:
    """
    This function is used in a post-processing step: 
    after the model training on data (X_sentences) this function goes through all the
    words and changes all the labels that match the regex pattern to label 'A'.
    """
    regex_mapping = group_words_into_sentences(pd.read_csv("../home/regex_mapping.csv"))
    merged = pd.merge(X_sentences, regex_mapping[["id", "sentence", "regex_rule"]], on=['id', 'sentence'], how='left')['regex_rule']   
    y_regex_pred = list(merged)

    result: List[List[str]] = []
    for regex_labels, crf_labels in zip(y_regex_pred, y_predictions):
        sentence_labels = []
        
        for regex_label, crf_label in zip(regex_labels, crf_labels):
            if regex_label == 'A':
                sentence_labels.append(regex_label)
            else:
                sentence_labels.append(crf_label)
        result.append(sentence_labels)
    
    return result


def satisfies_regex_rule(text: str, df: DataFrame) -> None:
    """
    This function takes a string and a pandas DataFrame including
    columns 'start' and 'end'. 
    As a result, the dataframe contains a new column 'regex_rule'
    that contains values indicating a word with offsets 'start',
    'end' in the text satisfies at least one of the predefined 
    regex rules.
    """
    regex_entities: List[Dict[str, Any]] = extract_entities(text)

    for entity in regex_entities:
        start = entity["start"]
        end = entity["end"]
        df.loc[(df["start"] >= start) & (df["end"] <= end), "regex_rule"] = 'A'
    df["regex_rule"] = df["regex_rule"].fillna('O')


def crf_feature_format(df: DataFrame) -> List[List[Dict[str, Any]]]:
    """
    This function takes a pandas DataFrame as an input and returns the
    provided data in a feature format suitable for the crf model.
    """
    result: List[List[Dict[str, Any]]] = []
    
    for _, row in df.iterrows():
        sentence: List[Dict[str, Any]] = []
        
        for i in range(len(row['word'])):
            word_dict: Dict[str, Any] = {}
            
            for col_name in df.columns:
                if isinstance(row[col_name], list) and len(row[col_name]) == len(row['word']):
                    word_dict[col_name] = row[col_name][i]
                else:
                    word_dict[col_name] = row[col_name]
            
            sentence.append(word_dict)
        result.append(sentence)
    return result


def robe_dataset(X_train: DataFrame, y_train: DataFrame) -> List[Dict[str, List[str]]]:
    """
    This function takes two pandas DataFrames as an input and returns the
    provided data in a format suitable for the robecz model.
    """
    result: List[Dict[str, List[str]]]= []
    
    for index, row in X_train.iterrows():
        word_label_pair = {
            'words': row['word'],
            'tags': [0 if x == 'O' else 1 for x in y_train[index]]
        }
        result.append(word_label_pair)
    return result


def add_basic_features(df: DataFrame, regex: bool) -> DataFrame:
    """
    This function returns a new dataframe containing basic features about each word in the dataframe
    that is provided as an input. It assumes the dataframe includes columns: 
    'word', 'sentence', 'start', 'end'.
    """
    for column in ["word", "sentence", "start", "end"]:
        assert column in df.columns
    
    data: DataFrame = df.copy()

    #regex label
    if regex and "id" in data.columns: 
        regex_mapping = group_words_into_sentences(pd.read_csv("../home/regex_mapping.csv"))
        
        for column in ["id", "sentence", "regex_rule"]:
            assert column in regex_mapping.columns
        
        data = pd.merge(df, regex_mapping[["id", "sentence", "regex_rule"]], 
                        on=["id", "sentence"], how='left')
    
    if "id" in df.columns:       
        # position within the text         
        temp = unroll_grouped_data(data[["id", "end"]])
        temp["text_length"] = temp.groupby("id")["end"].transform("max")
        temp = temp.drop_duplicates(subset='id')
        data = pd.merge(data, temp[['id', 'text_length']], on='id', how='left')
    else:
        # position within the text
        data["text_length"] = data.iloc[-1]['end'][-1]
    data["word.position"] = [[round(start / text_length, 2) for start in starts] 
                        for starts, text_length in zip(data["start"], data["text_length"])]
    data.drop(columns=["text_length"], inplace=True)

    # sufix
    data["word[-3:]"] = [[word[-3:] for word in sentence] for sentence in data["word"]]
    # data["word[-2:]"] = [[word[-2:] for word in sentence] for sentence in data["word"]]
    
    # POS tags
    tagger = Tagger("./models/czech-morfflex2.0-pdtc1.0-220710-pos_only.tagger")
    data["word.POS_tag"] = [list(list(tagger.tag(word))[0].tag for word in sentence) 
                            for sentence in data["word"]]

    # surface-level features
    data["word.len()"] =  [[len(word) for word in sentence] for sentence in data["word"]]
    data["word.istitle()"] = [[word.istitle() for word in sentence] for sentence in data["word"]]
    # data["word.isupper()"] = [[word.isupper() for word in sentence] for sentence in data["word"]]
    data["word.isdigit()"] = [[word.isdigit() for word in sentence] for sentence in data["word"]]
    data["bias"] = 1.0

    return data


def add_contextual_features(df: DataFrame, regex: bool, n: int) -> DataFrame:
    """
    This function returns a new dataframe containing contextual features about each word in the 
    dataframe that is provided as an input. It assumes the dataframe includes columns: 
    'word' and optionally 'regex_rule'.
    """
    assert "word" in df.columns
        
    tagger = Tagger("./models/czech-morfflex2.0-pdtc1.0-220710-pos_only.tagger")
    data: DataFrame = df.copy()

    if regex and "regex_rule" in data.columns:
        data[f'+{n}:word.regex_rule'] = [[sentence[i+n] if (i+n) < len(sentence) else "O" for i in range(len(sentence))] 
                                     for sentence in data["regex_rule"]]
        data[f'-{n}:word.regex_rule'] = [[sentence[i-n] if (i-n) >= 0 else "O" for i in range(len(sentence))] 
                                     for sentence in data["regex_rule"]]

    data[f'-{n}:word'] = [[sentence[i-n] if (i-n) >= 0 else "BOS" for i in range(len(sentence))] 
                          for sentence in data["word"]]
    data[f'-{n}:word[-3:]'] = [[sentence[i-n][-3:] if (i-n) >= 0 else "BOS" for i in range(len(sentence))] 
                                for sentence in data["word"]]
    #data[f'-{n}:word[-2:]'] = [[sentence[i-n][-2:] if (i-n) >= 0 else "BOS" for i in range(len(sentence))] 
                                #for sentence in data["word"]]
    data[f'-{n}:word.len()'] = [[len(sentence[i-n]) if (i-n) >= 0 else 0 for i in range(len(sentence))] 
                                for sentence in data["word"]]
    data[f'-{n}:word.istitle()'] = [[sentence[i-n].istitle() if (i-n) >= 0 else False for i in range(len(sentence))] 
                                    for sentence in data["word"]]
    data[f'-{n}:word.isdigit()'] = [[sentence[i-n].isdigit() if (i-n) >= 0 else False for i in range(len(sentence))] 
                                    for sentence in data["word"]]
    data[f'-{n}:word.POS_tag'] = [list(list(tagger.tag(sentence[i-n]))[0].tag if (i-n) >= 0 else 
                                       list(tagger.tag("."))[0].tag for i in range(len(sentence))) for sentence in data["word"]]

    data[f'+{n}:word'] = [[sentence[i+n] if (i+n) < len(sentence) else "EOS" for i in range(len(sentence))] 
                          for sentence in data["word"]]
    data[f'+{n}:word[-3:]'] = [[sentence[i+n][-3:] if (i+n) < len(sentence) else "EOS" for i in range(len(sentence))] 
                                for sentence in data["word"]]
    #data[f'+{n}:word[-2:]'] = [[sentence[i+n][-2:] if (i+n) < len(sentence) else "EOS" for i in range(len(sentence))] 
                                #for sentence in data["word"]]
    data[f'+{n}:word.len()'] = [[len(sentence[i+n]) if (i+n) < len(sentence) else 0 for i in range(len(sentence))] 
                                for sentence in data["word"]]
    data[f'+{n}:word.istitle()'] = [[sentence[i+n].istitle() if (i+n) < len(sentence) else False for i in range(len(sentence))] 
                                    for sentence in data["word"]]
    data[f'+{n}:word.isdigit()'] = [[sentence[i+n].isdigit() if (i+n) < len(sentence) else False for i in range(len(sentence))] 
                                    for sentence in data["word"]]
    data[f'+{n}:word.POS_tag'] = [list(list(tagger.tag(sentence[i+n]))[0].tag if (i+n) < len(sentence) else 
                                       list(tagger.tag("."))[0].tag for i in range(len(sentence))) for sentence in data["word"]]
    return data


def assign_sentence_id(df: DataFrame, sentences: List[str]) -> None:
    """
    This function takes a pandas DataFrame and a list of strings as an input. 
    It assumes the dataframe includes column 'word'.
    As a result, a 'sentence' column is added to the dataframe containing
    an index of the sentence that each word belongs to.
    """
    assert "word" in df.columns
    
    sentence_id: int = 0
    curr_sent_len: int = 0
    
    for i, word in enumerate(df["word"]):
        if (curr_sent_len + len(word)) <= len(sentences[sentence_id]) and word in sentences[sentence_id]:
            df.at[i, "sentence"] = sentence_id
            curr_sent_len += len(word)
        else:
            sentence_id += 1
            df.at[i, "sentence"] = sentence_id
            curr_sent_len = len(word)


def main():
    pass


if __name__ == '__main__':
    main()