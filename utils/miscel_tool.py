import pandas as pd
import matplotlib.pyplot as plt


def split_page_underscore(df_in):
    """inplace"""
    df_in['title'] = df_in['Page'].str.split('_').str[:-3].str.join('_')
    df_in[['project', 'device', 'agent']] = pd.DataFrame(
        df_in['Page'].str.split('_').str[-3:].to_list())
    return df_in


def extract_time_column_name(df_in):
    mask = df_in.columns.str.match('\d{4}-\d{2}-\d{2}')
    return df_in.columns[mask]


def plot_unique_value_mean(df_in, col_name, figsize=(12, 6), title=None, to_show=True, save_path=None):
    col_time = extract_time_column_name(df_in)
    unique_values = df_in[col_name].unique()

    plt.figure(figsize=figsize)

    for val in unique_values:
        mask_val = df_in[col_name] == val
        df_in.loc[mask_val, col_time].mean().plot(label=val)

    plt.legend()
    if title == None:
        title = f'Mean {col_name}'
    plt.title(title)
    plt.xticks(rotation=15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if to_show:
        plt.show()


def sort_by_attr_on_date(df_in, attr, on_date, topk=15):
    result = {}

    attr_subtypes = df_in[attr].unique()
    for subtype in attr_subtypes:
        mask_attr = df_in[attr] == subtype
        df_date_sort = df_in[mask_attr].sort_values(
            by=on_date, ascending=False)

        result[subtype] = df_date_sort.iloc[:topk]

    return result
