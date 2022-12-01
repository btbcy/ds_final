import argparse
import miscel


def get_args():
    parser = argparse.ArgumentParser(description='get summary from wiki')
    parser.add_argument('--start_index', type=int, default=0,
                        help='start index for parsing wiki summary')
    parser.add_argument('-o', '--out_path', type=str,
                        help='output file name')
    return parser.parse_args()


def main_parallelize():
    data_path = 'web_traffic_data/train_1.csv'
    df_en = miscel.read_en_data(data_path)
    df_en['summary'] = miscel.parallelize(df_en, miscel.write_summary)
    df_en.to_csv('summary_1.csv', index=False)


def main_sequantial(out_path, start_index: int):

    data_path = 'web_traffic_data/train_1.csv'
    df_en = miscel.read_en_data(data_path)

    print(f'start parsing from {start_index}')
    df_en['summary'] = miscel.NULL_VALUE
    df_summary = miscel.write_summary_sequential(
        df_en, out_path, start_index)

    print('finish')
    df_summary.to_csv('train_1_summary.csv', index=False)


if __name__ == "__main__":
    args = get_args()
    main_sequantial(args.out_path, args.start_index)
