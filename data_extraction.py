import pandas as pd
from io import StringIO
from urllib.request import urlopen
import sqlite3

def get_source(url):
    response = urlopen(url)
    html = response.read()
    return html.decode()

def get_tables_from_html(html):
    html_io = StringIO(html)
    return pd.read_html(html_io)

def save_tables_to_sqlite(tables, db_name):
    conn = sqlite3.connect(db_name)
    for name, table in tables.items():
        table.to_sql(name, conn, if_exists='replace', index=False)
    conn.close()

def process_table(table):
    table = table.iloc[::2]
    table = table.map(lambda x: str(x).replace('Último', '').replace('Anterior', '').replace('12 meses', '').replace('No ano', ''))
    table.iloc[:, 3] = table.iloc[:, 3].replace('  -', '0').str.replace(',', '.').astype(float)
    table.iloc[:, 4] = table.iloc[:, 4].replace('  -', '0').str.replace(',', '.').astype(float)
    
    return table

url = 'https://www.ibge.gov.br/indicadores#variacao-do-pib'
html_content = get_source(url)
tabelas = get_tables_from_html(html_content)

processed_tables = {f'data_{i+1}': process_table(tab) for i, tab in enumerate(tabelas)}
save_tables_to_sqlite(processed_tables, 'tabelas_ibge.db')
