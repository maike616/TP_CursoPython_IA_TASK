from urllib.request import urlopen  
from io import StringIO  
import pandas as pd  
import warnings  
import sqlite3  
import re  

class ProcessadorDeTabelasIBGE:
    def __init__(self, url, nome_banco):
        # Inicializa a classe com a URL da página do IBGE e o nome do banco de dados SQLite
        self.url = url
        self.nome_banco = nome_banco
        self.conteudo_html = None
        self.tabelas = None
        self.tabelas_processadas = None

    def obter_fonte(self):
        # Obtém o conteúdo HTML da página do IBGE
        resposta = urlopen(self.url)
        self.conteudo_html = resposta.read().decode()

    def obter_tabelas_do_html(self):
        # Lê as tabelas HTML contidas no conteúdo HTML
        html_io = StringIO(self.conteudo_html)
        self.tabelas = pd.read_html(html_io)

    @staticmethod
    def extrair_primeiro_numero(texto):
        # Extrai o primeiro número encontrado no texto, tratando formatos brasileiros de números
        correspondencia = re.search(r'\d{1,3}(?:\.\d{3})*(?:,\d+)?', texto)
        if correspondencia:
            numero_str = correspondencia.group().replace('.', '').replace(',', '.')
            return float(numero_str)
        return 0.0

    @staticmethod
    def preprocessamento_dados_tabela(tabela):
        # Processa a tabela removendo linhas alternadas e limpando valores
        tabela = tabela.iloc[::2] 
        tabela = tabela.applymap(lambda x: str(x).replace('Último', '').replace('Anterior', '').replace('12 meses', '').replace('No ano', ''))
        
        # Aplica a extração de números nas colunas 1 e 2
        tabela.iloc[:, 1] = tabela.iloc[:, 1].apply(ProcessadorDeTabelasIBGE.extrair_primeiro_numero)
        tabela.iloc[:, 2] = tabela.iloc[:, 2].apply(ProcessadorDeTabelasIBGE.extrair_primeiro_numero)
        
        # Converte valores das colunas 3 e 4 para float, tratando ' -' como 0
        tabela.iloc[:, 3] = tabela.iloc[:, 3].replace('  -', '0').apply(lambda x: str(x).replace('.', '').replace(',', '.')).astype(float)
        tabela.iloc[:, 4] = tabela.iloc[:, 4].replace('  -', '0').apply(lambda x: str(x).replace('.', '').replace(',', '.')).astype(float)
        return tabela

    def processar_tabelas(self):
        # Processa todas as tabelas lidas do HTML e as organiza por chaves temáticas
        warnings.filterwarnings('ignore')  
        chaves = ['Economia', 'Social', 'Agropecuario']
        self.tabelas_processadas = {chaves[i]: self.preprocessamento_dados_tabela(tab) for i, tab in enumerate(self.tabelas)}

    def salvar_tabelas_no_sqlite(self):
        # Salva as tabelas processadas em um banco de dados SQLite
        conexao = sqlite3.connect(self.nome_banco)
        for nome, tabela in self.tabelas_processadas.items():
            tabela.to_sql(nome, conexao, if_exists='replace', index=False)
        conexao.close()

    def executar(self):
        # Executa todo o processo: obter fonte, extrair e processar tabelas, e salvar no banco de dados
        self.obter_fonte()
        self.obter_tabelas_do_html()
        self.processar_tabelas()
        self.salvar_tabelas_no_sqlite()

# URL da página do IBGE de onde os dados são obtidos
url = 'https://www.ibge.gov.br/indicadores#variacao-do-pib'
nome_banco = 'tabelas_ibge.db'
processador = ProcessadorDeTabelasIBGE(url, nome_banco)
# Executa o processamento
processador.executar()
