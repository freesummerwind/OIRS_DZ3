import sqlite3
import csv
from config import path_to_sqlite_db, \
    group_info_columns, post_columns_csv, post_columns_db
import pandas as pd


def create_table(cur, table_name, columns_str):
    print(columns_str)
    cur.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_str});")


def insert_data_from_csv(cur, table_name, columns_list_csv, columns_str_db, csv_file_path):
    with open(csv_file_path, 'r') as fin:
        dr = csv.DictReader(fin)
        to_db = [[data[col] for col in columns_list_csv] for data in dr]
    cur.executemany(f"INSERT INTO {table_name} (" + columns_str_db +
                    ") VALUES (" + ', '.join(['?'] * len(columns_list_csv)) + ");", to_db)


def create_tables_insert_data(cur, path_group_csv, path_post_csv):
    group_columns_str = ', '.join(group_info_columns)
    create_table(cur, 'vk_group_info', group_columns_str)
    insert_data_from_csv(cur, 'vk_group_info', group_info_columns, group_columns_str, path_group_csv)

    post_columns_str = ', '.join(post_columns_db)
    create_table(cur, 'vk_post_info', post_columns_str)
    insert_data_from_csv(cur, 'vk_post_info', post_columns_csv, post_columns_str, path_post_csv)


def get_last_rows(con, table_name):
    df = pd.read_sql_query(
        f'SELECT * FROM {table_name} LIMIT 10 OFFSET (SELECT COUNT(*) FROM {table_name})-10', con
    )
    return df


def insert_data_into_table(cur, table_name, columns_list_csv, columns_str_db, df):
    data = df.values
    print(data)
    cur.executemany(f"INSERT INTO {table_name} (" + columns_str_db +
                    ") VALUES (" + ', '.join(['?'] * len(columns_list_csv)) + ");", data)


def insert_data(con, group_df, post_df):
    cur = con.cursor()
    group_columns_str = ', '.join(group_info_columns)
    insert_data_into_table(cur, 'vk_group_info', group_info_columns, group_columns_str, group_df)
    con.commit()

    post_columns_str = ', '.join(post_columns_db)
    print(post_columns_csv, post_columns_str)
    insert_data_into_table(cur, 'vk_post_info', post_columns_csv, post_columns_str, post_df)
    con.commit()


if __name__ == '__main__':
    connection = sqlite3.connect(path_to_sqlite_db)
    cursor = connection.cursor()

    print(get_last_rows(connection, 'vk_group_info'))

    print(get_last_rows(connection, 'vk_post_info'))
