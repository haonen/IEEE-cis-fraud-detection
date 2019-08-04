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

    def compute_average_value(self, set_label, feature, group_by_features, low_dt=None, high_dt=None):
        """
        Compute average of `feature` group by `group_by_features`.
        `group_by_features` is a list of features.

        Example: compute the transaction amount according to card4 and card6
        ```
        cur = datasource.compute_average_value("train", "transactionamt", ["card4", "card6"])
        ```

        If `low_dt` and `high_dt` are provided, only the data of transactiondt fall in [low_dt, high_dt] will be selected.
        """
        assert ((low_dt is None) and (high_dt is None)) or ((low_dt is not None) and (high_dt is not None))
        tt = set_label + '_transaction'
        it = set_label + '_identity'
        feature = "foo." + feature
        group_by_features = ["foo." + item for item in group_by_features]
        group_by_features_str = ", ".join(group_by_features)
        view_table_sub = "(SELECT * FROM {0} JOIN {1} USING (transactionid))".format(tt, it)
        sql = "SELECT " + group_by_features_str + ", AVG("+ feature + ") FROM "
        sql += view_table_sub + " AS foo"
        if low_dt is not None:
            assert low_dt <= high_dt
            sql += " WHERE foo.transactiondt>={0} AND foo.transactiondt<{1}".format(low_dt, high_dt)
        sql += " GROUP BY " + group_by_features_str
        sql +=";"
        cur = self.dbinstance.execure_sql(sql)
        return cur

    def set_impute_table_value(self, card4, card6, feature_name, impute_value):
        """
        set impute_table value of `feature_name` to `impute_value`.
        if the record for (card4, card6) does not exist, it will create one.
        """
        cur = self.dbinstance.execure_sql("select count(*) from impute_table where card4='"+card4+"' and card6='"+card6+"';")
        record_num = cur.fetchone()[0]
        if record_num == 0:
            sql = "insert into impute_table(card4, card6, {0}) values ('{1}', '{2}', {3});"
            sql = sql.format(feature_name, card4, card6, impute_value)
            self.dbinstance.execure_sql(sql, True)
        else:
            sql = "update impute_table set {0}={1} where card4='{2}' and card6='{3}';"
            sql = sql.format(feature_name, impute_value, card4, card6)
            self.dbinstance.execure_sql(sql, True)
    def read_impute_table_value(self, card4, card6, feature_names=None):
        """
        read the record in impute table of (card4, card6) does not exist.
        return a cursor
        `feature_names` is the list of all features you want to retrieve. None means all features.
        """
        if feature_names == None:
            feature_names = "*"
        else:
            feature_names = ", ".join(feature_names)
        cur = self.dbinstance.execure_sql("select {0} from impute_table where card4='{1}' and card6='{2}';".format(feature_names, card4, card6))
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
    cur =datasource.compute_average_value("train", "transactionamt", ["card4", "card6"], low_dt=87000, high_dt=89000)
    df = cursor_to_dataframe(cur)
    print(df)
    datasource.set_impute_table_value('visa', 'debit', 'transactionamt', 11)
    datasource.set_impute_table_value('visa', 'credit', 'transactionamt', 20)
    cur = datasource.read_impute_table_value('visa', 'credit')
    print(cursor_to_dataframe(cur))
