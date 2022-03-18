#################################################################################################
# Sauvegarde des données dans une postgreSQL
#################################################################################################
import psycopg2
from pprint import pprint


class DataBaseConnect:
    def __init__(self):
        try:
            self.conn = psycopg2.connect("dbname=ComMindDB user=postgres password=root")
            self.conn.autocommit = True
            self.cur = self.conn.cursor()
        except:
            pprint("Echec de la connexion à la base de donnée")

    def insertNewProduct(self, **args):
        colums_name = tuple(args['args'].keys())
        insert_values = tuple(args['args'].values())
        insert_script = f"INSERT INTO lefouineur_product ({', '.join(colums_name)}) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        self.cur.execute(insert_script, insert_values)
        print('Données rajoutées avec succès')
        pprint(args['args'])

    def insertNewProductRemark(self, **args):
        colums_name = tuple(args['args'].keys())
        insert_values = tuple(args['args'].values())
        insert_script = f"INSERT INTO lefouineur_productremark ({', '.join(colums_name)}) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        self.cur.execute(insert_script, insert_values)
        print('Données rajoutées avec succès')
        pprint(args['args'])

    # def update_product(self):
    #     update_command = "UPDATE table SET colum=value WHERE id=1"

    def querySearch(self, product_key):
        try:
            self.cur.execute(f"SELECT * FROM lefouineur_product WHERE product_business_key = '{product_key}'")
            # Retourne le premier element s'il existe sinon retourne un query vide
            return query[0] if (query := self.cur.fetchall()) else query
        except:
            return None

    def closedCon(self):
        self.conn.close()


def colToSave(df, i):
    columns = df.columns.tolist()
    excluded_col = ['Document_No', 'Text', 'clear_comment', 'comments_count']
    for col in excluded_col:
        columns.remove(col)
    return {column: df[column][i] for column in columns}


def save_data(frame):
    shape_lenght = frame.shape[0]
    bd = DataBaseConnect()
    for i in range(shape_lenght):

        data_dict = colToSave(frame, i)

        product = bd.querySearch(product_key=data_dict['_id'])

        if not product:
            bd.insertNewProduct(args={
                    'product_business_key': data_dict['_id'], 'title': data_dict['product_title'],
                    'product_url': data_dict['product_url'], 'image_url': data_dict['image_url'],
                    'product_rating': data_dict['rating'], 'nb_reviewers': int(data_dict['nb_reviewers']),
                    'price': 0.0
                })
            product = bd.querySearch(product_key=data_dict['_id'])

        bd.insertNewProductRemark(args={
                'remark': data_dict['comment'], 'date': data_dict['date'], 'region': data_dict['region'],
                'keyword': data_dict['Keywords'], 'topic_perc_contrib': float(data_dict['Topic_Perc_Contrib']),
                'dominant_topic': int(data_dict['Dominant_Topic']), 'polarity': float(data_dict['polarity']),
                'subjectivity': float(data_dict['subjectivity']), 'product_id': product[0]
            })
