# Projeto Final - Programação com Python

## Universidade Federal do Amazonas – UFAM
**Projeto SUPER**

**Disciplina**: Programação com Python  
**Professor**: Francisco Januário  

### Descrição do Projeto
Este repositório contém o projeto final da Equipe 4 para a disciplina de Programação com Python, oferecida pela Universidade Federal do Amazonas (UFAM) no Projeto SUPER. O objetivo do projeto é implementar um programa que lê dados diretamente de um site da Web, organiza esses dados em um DataFrame e armazena em um banco de dados SQLite. Em seguida, outro programa é implementado para ler os dados do banco de dados, gerar um DataFrame e mostrar os dados em uma interface gráfica de usuário (GUI) usando Tkinter, exibindo tabelas e gráficos.

### Equipe 4: Painel de indicadores de variação do PIB
URL do site: [Painel de indicadores da variação do PIB - IBGE](https://www.ibge.gov.br/indicadores#variacao-do-pib)

### Funcionalidades do Projeto

1. **Coleta de Dados**
   - Leitura dos dados diretamente da página do IBGE sobre os indicadores.
   - Organização dos dados em um DataFrame utilizando a biblioteca pandas.
   - Armazenamento dos dados em um banco de dados SQLite.

2. **Visualização de Dados**
   - Leitura dos dados armazenados no banco de dados SQLite.
   - Geração de DataFrames a partir dos dados lidos.
   - Implementação de uma interface gráfica de usuário (GUI) utilizando Tkinter.
   - Exibição de tabelas com os dados do DataFrame na GUI.
   - Plotagem de gráficos utilizando Matplotlib para visualizar as séries temporais do DataFrame.

### Estrutura do Repositório

- `data_extraction.py`: Script responsável pela extração dos dados do site e armazenamento no banco de dados SQLite.
- `data_visualization.py`: Script responsável pela leitura dos dados do banco de dados, geração dos DataFrames e exibição dos dados em uma interface gráfica.
- `README.md`: Este arquivo de descrição do repositório.
- `requirements.txt`: Lista de bibliotecas necessárias para executar os scripts.
  
### Como Executar

1. **Configuração do Ambiente**
   - Certifique-se de ter o Python instalado em sua máquina.
   - Instale as bibliotecas necessárias utilizando o comando:
     ```bash
     pip install -r requirements.txt
     ```

2. **Extração de Dados**
   - Execute o script `data_extraction.py` para extrair os dados do site e armazená-los no banco de dados SQLite:
     ```bash
     python data_extraction.py
     ```

3. **Visualização de Dados**
   - Execute o script `data_visualization.py` para carregar os dados do banco de dados e exibi-los na interface gráfica:
     ```bash
     python data_visualization.py
     ```

### Exemplo de Uso

- Após executar `data_visualization.py`, uma janela GUI será aberta mostrando uma tabela com os dados do PIB e gráficos ilustrando a variação do PIB ao longo do tempo.

![image](https://github.com/maike616/TP_CursoPython_IA_TASK/assets/32426980/f83d9c26-afb0-496e-9c36-16741d6d6843)

![image](https://github.com/maike616/TP_CursoPython_IA_TASK/assets/32426980/a03593a6-be2e-4afd-9603-e0fa990155b3)


### Links Úteis
- [Como plotar um gráfico no Tkinter](https://www.pythontutorial.net/tkinter/tkinter-matplotlib/)
- [Criar tabela usando o Tkinter](https://www.geeksforgeeks.org/create-table-using-tkinter/)

### Contribuição
Para contribuir com este projeto, faça um fork do repositório, crie um branch para sua feature ou correção, e envie um pull request com suas alterações.

---

Equipe 4: [Marcos Augusto e Aline Lima]

