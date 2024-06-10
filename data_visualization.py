import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sqlite3
import tkinter as tk
from tkinter import ttk

class CarregadorSQLite:
    def __init__(self, nome_banco_dados):
        self.nome_banco_dados = nome_banco_dados

    def carregar_tabelas(self):
        # Conecta ao banco de dados SQLite
        conn = sqlite3.connect(self.nome_banco_dados)
        cursor = conn.cursor()
        # Obtém os nomes das tabelas no banco de dados
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tabelas = cursor.fetchall()

        # Carrega os dados de cada tabela em um DataFrame do pandas
        dataframes = {nome_tabela[0]: pd.read_sql_query(f"SELECT * FROM {nome_tabela[0]}", conn) for nome_tabela in tabelas}
        conn.close()
        return dataframes

class VisualizadorDataFrame(tk.Tk):
    def __init__(self, dataframes):
        super().__init__()
        self.title("Trabalho Final do Curso de Python")

        # Configuração para maximizar a janela
        self.state('zoomed')
        
        self.estilo = ttk.Style(self)
        self.estilo.theme_use('clam')
        self._configurar_estilos()

        self.dataframes = dataframes
        self.dataframe_atual = self.dataframes.get('Economia', list(dataframes.values())[0])
        self.canvas = None

        self.criar_widgets()

    def _configurar_estilos(self):
        # Configurações de estilo para os widgets do tkinter
        self.estilo.configure('TFrame', background='#f0f0f0')
        self.estilo.configure('TButton', background='#007acc', foreground='white', padding=6, font=('Arial', 10, 'bold'))
        self.estilo.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.estilo.configure('TCombobox', padding=6, font=('Arial', 10))
        self.estilo.configure('Treeview', font=('Arial', 10), rowheight=25)

    def criar_widgets(self):
        # Cria o container principal
        container = ttk.Frame(self)
        container.pack(fill='both', expand=True, padx=10, pady=10)

        # Frame para os controles
        frame_controles = ttk.Frame(container)
        frame_controles.pack(fill='x', pady=10)

        # Combobox para selecionar a tabela
        self.seletor_tabelas = ttk.Combobox(frame_controles, values=list(self.dataframes.keys()))
        self.seletor_tabelas.set('Economia' if 'Economia' in self.dataframes else list(self.dataframes.keys())[0])
        self.seletor_tabelas.pack(side='left', padx=5)
        self.seletor_tabelas.bind("<<ComboboxSelected>>", self.carregar_tabela)

        # Botão para gerar gráfico
        self.botao_grafico = ttk.Button(frame_controles, text="Gráfico", command=self.plotar_grafico)
        self.botao_grafico.pack(side='left', padx=5)

        # Frame para a Treeview
        self.frame_tree = ttk.Frame(container)
        self.frame_tree.pack(expand=True, fill='both')

        # Scrollbar para a Treeview
        self.scroll_tree = ttk.Scrollbar(self.frame_tree, orient='vertical')
        self.scroll_tree.pack(side='right', fill='y')

        # Treeview para exibir os dados do DataFrame
        self.tree = ttk.Treeview(self.frame_tree, yscrollcommand=self.scroll_tree.set)
        self.tree.pack(expand=True, fill='both')
        self.scroll_tree.config(command=self.tree.yview)

        # Carrega a tabela inicial
        self.carregar_tabela()

    def carregar_tabela(self, event=None):
        # Obtém o nome da tabela selecionada
        nome_tabela = self.seletor_tabelas.get()
        self.dataframe_atual = self.dataframes[nome_tabela]

        # Limpa os dados atuais da Treeview
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(self.dataframe_atual.columns)
        self.tree["show"] = "headings"

        # Define os cabeçalhos e largura das colunas
        for coluna in self.tree["columns"]:
            self.tree.heading(coluna, text=coluna)
            self.tree.column(coluna, width=100, stretch=tk.YES)

        # Insere os dados na Treeview
        for _, linha in self.dataframe_atual.iterrows():
            self.tree.insert("", "end", values=list(linha))

        # Ajusta a largura das colunas
        self._ajustar_largura_colunas()

    def _ajustar_largura_colunas(self):
        # Ajusta a largura das colunas para se adequar ao conteúdo
        for coluna in self.tree["columns"]:
            largura_max = max(len(str(valor)) for valor in self.dataframe_atual[coluna])
            self.tree.column(coluna, width=largura_max * 10)

    def plotar_grafico(self):
        # Limpa o canvas existente, se houver
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
            self.canvas.get_tk_widget().destroy()

        # Cria uma nova figura e um eixo para o gráfico
        fig, ax = plt.subplots()
        colunas = list(self.dataframe_atual.columns)
        coluna_x = colunas[0]
        colunas_y = colunas[1:5]

        # Converte as colunas selecionadas para numérico, se possível
        self.dataframe_atual[colunas_y] = self.dataframe_atual[colunas_y].apply(pd.to_numeric, errors='coerce')

        marcadores = ['o', 's', '^', 'D']  # Diferentes símbolos para as linhas
        for i, coluna_y in enumerate(colunas_y):
            if pd.api.types.is_numeric_dtype(self.dataframe_atual[coluna_y]):
                ax.plot(self.dataframe_atual[coluna_x], self.dataframe_atual[coluna_y], label=coluna_y, marker=marcadores[i], linestyle='-', markersize=5)
        ax.set_ylabel("Valores")
        ax.set_title(coluna_x)
        ax.legend()

        # Cria um canvas para exibir o gráfico
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(expand=True, fill='both')


if __name__ == "__main__":
    # Nome do banco de dados SQLite
    nome_banco_dados = 'tabelas_ibge.db'
    carregador = CarregadorSQLite(nome_banco_dados)
    dataframes = carregador.carregar_tabelas()
    
    # Substituindo a coluna 0 do dataframe 'Agropecuario' com novos valores
    novos_valores = ["Leite", "Ovos", "Bovinos", "Suínos", "Galináceos", "Banana", "Café", "Cana-de-açúcar", "Laranja", "Milho", "Soja"]
    dataframes['Agropecuario'].iloc[:, 0] = novos_valores
    
    # Remove a linha 6 do dataframe 'Economia'
    dataframes['Economia'] = dataframes['Economia'].drop(dataframes['Economia'].index[6])

    # Inicializa e executa o visualizador de DataFrame
    app = VisualizadorDataFrame(dataframes)
    app.mainloop()
