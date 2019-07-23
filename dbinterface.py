"""
SQL statement builder and abstract of the database
"""
import psycopg2
from config import config
from utils import *
import os

def create_table_from_csv_sql(csv_file, non_number_column_pattern, table_name):
    """
    Return a SQL statement according to the `csv_file` to create the table.
    Define non-number columns by `non_number_column_pattern`.
    """
    header = read_header(csv_file)
    header_with_type = []
    for item in header:
        if match_in_pattern_list(non_number_column_pattern, item):
            header_with_type.append((item, 'varchar(50)'))
        else:
            header_with_type.append((item, 'real'))
    assert header_with_type[0][1] == 'real', 'Primary key must be number'
    header_with_type[0] = (header[0], 'int NOT NULL')
    sql_statement = "CREATE TABLE " + table_name + "\n(\n"
    for col, dbt in header_with_type:
        sql_statement += "\t" + col + " " + dbt + ",\n"
    sql_statement += "\tPRIMARY KEY(" + header[0] + ")\n);"
    return sql_statement


class DBInterface:
    connection = None
    cur = None
    def __init__(self, user, password, host, port, database):
        self.connection = psycopg2.connect(user=user, 
                password=password, 
                host=host, 
                port=port, 
                database=database)
        self.cur = self.connection.cursor()
    
    def __del__(self):
        self.connection.close()
    
    def build_table_from_csv(self, csv_file, non_number_column_pattern, table_name):
        # drop existing table
        sql_statement = "DROP TABLE " + table_name + ";"
        try:
            self.cur.execute(sql_statement)
        except psycopg2.errors.UndefinedTable:
            self.connection.rollback()
        # create table
        sql_statement = create_table_from_csv_sql(csv_file, non_number_column_pattern, table_name)
        self.cur.execute(sql_statement)
        # copy data
        header = read_header(csv_file)
        f = open(csv_file)
        f.readline()
        self.cur.copy_from(f, table_name, columns=header, sep=',', null="")
        self.connection.commit()

class DataSource:
    """
    Abstract of data source.
    """
    TRANSACTION_NON_NUMBER_PATTERN = ['ProductCD', 'card\d', 'addr\d', 'M\d', '.*domain']
    IDENTITY_NON_NUMBER_PATTERN = ['Device.*', 'id_1[^01]', 'id_2.', 'id_3.']
    def __init__(self):
        self.dbinstance = DBInterface(config.USERNAME, config.PASSWORD, '127.0.0.1', '5432', 'ieeefraud')
    
    def load_data_to_db(self, path):
        """
        Load data from csv files to database. 
        `path` is the path that stores all csv files.
        """
        table_names = ['train_transaction', 'train_identity', 'test_transaction', 'test_identity']
        for table_name in table_names:
            pat = self.TRANSACTION_NON_NUMBER_PATTERN if 'transaction' in table_name else self.IDENTITY_NON_NUMBER_PATTERN
            print("Loading table: " + table_name)
            fn = os.path.join(path, table_name + '.csv')
            self.dbinstance.build_table_from_csv(fn, pat, table_name)
            print("Loaded table " + table_name)

if __name__ == "__main__":
    datasource = DataSource()
    datasource.load_data_to_db('.')
