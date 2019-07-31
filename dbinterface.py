"""
SQL statement builder and abstract of the database
"""
import psycopg2
from config import config
from utils import *
import os
import pandas

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
    def __init__(self, user, password, host, port, database):
        self.connection = psycopg2.connect(user=user, 
                password=password, 
                host=host, 
                port=port, 
                database=database)
    
    def __del__(self):
        self.connection.close()
    
    def build_table_from_csv(self, csv_file, non_number_column_pattern, table_name):
        # drop existing table
        sql_statement = "DROP TABLE " + table_name + ";"
        cur = self.connection.cursor()
        try:
            cur.execute(sql_statement)
        except psycopg2.errors.UndefinedTable:
            self.connection.rollback()
        # create table
        sql_statement = create_table_from_csv_sql(csv_file, non_number_column_pattern, table_name)
        cur.execute(sql_statement)
        # copy data
        header = read_header(csv_file)
        f = open(csv_file)
        f.readline()
        cur.copy_from(f, table_name, columns=header, sep=',', null="")
        self.connection.commit()
    
    def get_cursor(self):
        """
        Return a cursor of the database.
        """
        return self.connection.cursor()
 
    def execure_sql(self, sql_statement, to_write=False):
        """
        Execute an SQL statement.
        Set `to_write` to True if you want to write values
        If `to_write` is False, return a cursor of data
        Else return None.
        """
        cur = self.connection.cursor()
        cur.execute(sql_statement)
        if to_write:
            self.connection.commit()
            cur.close()
        else:
            return cur
       
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

    def select_data_by_transactiondt(self, set_label, low, high, feature_list=None):
        """
        Select data from a set where transactionDT is in [low, high).
        `set_label` can be `train` or `test`. 
        If `feature_list` is None, all columns will be retrieved;

        Return a cursor object of the selected data. 
        """
        tt = set_label + '_transaction'
        it = set_label + '_identity'
        stat = "SELECT {0} FROM {1}, {2} WHERE {1}.transactionid={2}.transactionid \
                AND {1}.transactiondt>={3} AND {1}.transactiondt<{4};"
        
        if feature_list is not None:
            # transactionid is an ambigious term which appears in both table
            feature_list = [item if item!='transactionid' else tt+'.' + item for item in feature_list]
            feature_list= ', '.join(feature_list)
        else:
            feature_list = "*"
        stat = stat.format(feature_list, tt, it, low, high)
        cur = self.dbinstance.get_cursor();
        cur.execute(stat)
        return cur

    def compute_average_value(self, set_label, feature, group_by_features):
        """
        Compute average of `feature` group by `group_by_features`.
        `group_by_features` is a list of features.

        Example: compute the transaction amount according to card4 and card6
        ```
        cur = datasource.compute_average_value("train", "transactionamt", ["card4", "card6"])
        ```
        """
        tt = set_label + '_transaction'
        it = set_label + '_identity'
        feature = "foo." + feature
        group_by_features = ["foo." + item for item in group_by_features]
        group_by_features_str = ", ".join(group_by_features)
        view_table_sub = "(SELECT * FROM {0} JOIN {1} USING (transactionid))".format(tt, it)
        sql = "SELECT " + group_by_features_str + ", AVG("+ feature + ") FROM "
        sql += view_table_sub + " AS foo GROUP BY " + group_by_features_str
        sql +=";"
        cur = self.dbinstance.execure_sql(sql)
        return cur

 
def cursor_to_dataframe(cur):
    """
    Transfer a database cursor into a pandas DataFrame.
    """
    description = cur.description
    column_names = [item.name for item in description]
    data = cur.fetchall()
    df = pandas.DataFrame(data, columns=column_names)
    cur.close()
    return df

if __name__ == "__main__":
    datasource = DataSource()
    # datasource.load_data_to_db('.')
    # cur = datasource.select_data_by_transactiondt('train', 0, 1000000000, ['transactionid', 'card4', 'card6', 'devicetype', 'id_01', 'id_02', 'isfraud'])
    cur =datasource.compute_average_value("train", "transactionamt", ["card4", "card6"])
    df = cursor_to_dataframe(cur)
    print(df)
