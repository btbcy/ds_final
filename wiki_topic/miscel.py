from multiprocessing import Pool
from functools import partial
import time
import logging

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from torch.utils.data import Dataset
from transformers import pipeline

import wikipedia


NULL_VALUE = -1

LIST_OF_TOPICS = [
    "Art and culture",
    "Geography and places",
    "Health and fitness",
    "History and events",
    "Mathematics and abstractions",
    "Natural sciences and nature",
    "People and self",
    "Philosophy and thinking",
    "Religion and spirituality",
    "Social sciences and society",
    "Technology and applied sciences",
]

LABEL_TOPIC_MAP = {
    "Art": "Art and culture",
    "culture": "Art and culture",
    "Geography": "Geography and places",
    "places": "Geography and places",
    "Health": "Health and fitness",
    "fitness": "Health and fitness",
    "History": "History and events",
    "events": "History and events",
    "Mathematics": "Mathematics and abstractions",
    "abstractions": "Mathematics and abstractions",
    "Natural sciences": "Natural sciences and nature",
    "nature": "Natural sciences and nature",
    "People": "People and self",
    "self": "People and self",
    "Philosophy": "Philosophy and thinking",
    "thinking": "Philosophy and thinking",
    "Religion": "Religion and spirituality",
    "spirituality": "Religion and spirituality",
    "Social sciences": "Social sciences and society",
    "society": "Social sciences and society",
    "Technology": "Technology and applied sciences",
    "applied sciences": "Technology and applied sciences",
}

CANDIDATE_LABELS = [
    "Art",
    "culture",
    "Geography",
    "places",
    "Health",
    "fitness",
    "History",
    "events",
    "Mathematics",
    "abstractions",
    "Natural sciences",
    "nature",
    "People",
    "self",
    "Philosophy",
    "thinking",
    "Religion",
    "spirituality",
    "Social sciences",
    "society",
    "Technology",
    "applied sciences",
]

wikipedia.set_lang('en')


# multiprocess
def parallelize(data, func, num_of_processes=4):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


def run_on_subset(func, data_subset):
    return data_subset.apply(func, axis=1)


def parallelize_on_rows(data, func, num_of_processes=8):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)

# dataframe


def read_en_data(data_path):
    df = pd.read_csv(data_path, usecols=['Page'])
    en_pattern = "_en\."
    df_en = df.loc[df['Page'].str.contains(en_pattern)].copy()
    df_en['title'] = df_en['Page'].str.split(en_pattern).str[0]
    return df_en


def extract_parentheses(df_in):
    df_out = df_in.copy()
    df_out['parentheses'] = df_in['title'].str.extract(r'\((.*)\)')
    df_out.loc[df_out['parentheses'].isnull(), 'parentheses'] = NULL_VALUE
    df_out['parentheses'] = df_out['parentheses'].str.lower()
    df_out['parentheses'] = df_out['parentheses'].fillna(f'{NULL_VALUE}')
    return df_out

# wiki


def write_summary(row_in):
    summary_wiki = NULL_VALUE
    try:
        summary_wiki = get_page(row_in['title'])
    except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
        pass
    time.sleep(0.01)
    return summary_wiki


def write_summary_sequential(df_in, temp_name, start_index):
    df_out = df_in.copy()
    for step, idx in tqdm(enumerate(df_in.index[start_index:], start=start_index)):
        try:
            df_out.at[idx, 'summary'] = get_page(df_in.at[idx, 'title'])
        except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
            pass
        if step % 100 == 0:
            print(f'step={step}')
            df_out.to_csv(temp_name, index=False)
    return df_out


def get_page(title: str, num_of_sentence=1):
    one_page = wikipedia.page(title)
    summary_all = one_page.summary
    summary_extract = summary_all.split(". ")[:num_of_sentence]
    return ". ".join(summary_extract)

# transformer


class SummaryDataset(Dataset):
    def __init__(self, path_to_df: str) -> None:
        super().__init__()
        self.df = pd.read_csv(path_to_df)
        self.summary = self.df['summary'].to_list()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.summary[idx]


def create_model():
    classifier = pipeline("zero-shot-classification",
                          model="facebook/bart-large-mnli", device=0)
    return classifier


def predict_all(classifier, summary_set, batch_size=8):
    result = []
    score = []
    # not compatible with tqdm
    idx = 1
    for pred in classifier(summary_set, batch_size=batch_size,
                           candidate_labels=CANDIDATE_LABELS):
        result.append(pred['labels'][0])
        score.append(pred['scores'][0])
        if idx % 100 == 0:
            print(idx)
        idx += 1
    return result, score


def predict_one(classifier, sequence, verbose=True, top_k=5):
    result = classifier(sequence, CANDIDATE_LABELS)
    if verbose:
        for lbl, prob in zip(result['labels'][:top_k], result['scores'][:top_k]):
            print(f'{lbl}: {prob}')
    return result['labels'], result['scores']
