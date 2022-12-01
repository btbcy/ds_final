import sys
import pandas as pd

NULL_CLASS_VALUE = '-1'


def apply_all_rules(df_in):
    df_out = df_in.copy()

    rules = get_rules()
    rules.append(title_olympics)
    for rule_func in rules:
        df_out = rule_func(df_out)

    return df_out


def get_rules():
    curr = sys.modules[__name__]
    rule_name = [func for func in dir(curr)
                 if callable(getattr(curr, func)) and func.startswith('rule_')]
    rule_func = [getattr(curr, func) for func in rule_name]
    return rule_func


def rule_art(df_in):
    keywords = 'album|dance|song|soundtrack|book'
    return _contains_keywords(df_in, keywords, 'Art')


def rule_film(df_in):
    key_words = 'film|seasons?_\d+|tv_series|show'
    return _contains_keywords(df_in, key_words, 'Art')


def rule_acgn(df_in):
    keywords = 'comics|video_game|novel|^game$'
    return _contains_keywords(df_in, keywords, 'Art')


def rule_people(df_in):
    keywords = 'person|actor|actress|.*er$|.*cian$|.*ist|band'
    return _contains_keywords(df_in, keywords, 'People')


def rule_programming(df_in):
    keywords = 'programming_language|java|python'
    return _contains_keywords(df_in, keywords, 'Technology')


def rule_science(df_in):
    keywords = 'genus'
    return _contains_keywords(df_in, keywords, 'Natural sciences')


def title_olympics(df_in):
    df_out = df_in.copy()
    mask_olympics = df_out['lower_title'].str.contains('olympics')
    mask_null = df_out['class'] == '-1'
    df_out.loc[mask_olympics & mask_null, 'class'] = 'events'
    return df_out


def _contains_keywords(df_in, keywords, classname):
    df_out = df_in.copy()
    mask_keywords = df_out['parentheses'].str.contains(keywords)
    df_out.loc[mask_keywords, 'class'] = classname
    df_out.loc[mask_keywords, 'score'] = -1
    return df_out
