#Importando os frameworks

import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
import torch
from torch import nn
from torch import optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader, random_split
import random
from sqlalchemy import create_engine
from datetime import datetime
from sqlalchemy import text
from unidecode import unidecode
from fuzzywuzzy import process
from azure.storage.blob import BlobServiceClient
import time
import os


# Carregando as tabelas do SQL Server

# Dados de conexão
server = os.getenv('server')
database = os.getenv('database')
database2 = os.getenv('database2')
username = os.getenv('username')
password = os.getenv('password')
driver = 'ODBC Driver 17 for SQL Server'

# String de conexão
connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}'

# Criar engine
engine = create_engine(connection_string)

# Executar a consulta e carregar os dados em um DataFrame
try:
    with engine.connect() as conn:
        query = """SELECT * FROM xxdk_cust_inv_tax_detail WHERE trx_date >= '2022-05-01';"""
        query2 = """SELECT 
                            organization_name,
                            entity_name,
                            cnpj,
                            invoice_type,
                            client_location,
                            cost_center,
                            invoice_num,
                            invoice_date
                            
                    FROM xxdk_supplier_inv_cnpj WHERE invoice_date >= '2023-05-01';"""
        Customer_original = pd.read_sql_query(query, conn) 
        Supplier_original = pd.read_sql_query(query2, conn)
        
except Exception as e:
    print(f"Erro ao executar a consulta: {e}")

# String de conexão
connection_string2 = f'mssql+pyodbc://{username}:{password}@{server}/{database2}?driver={driver}'

# Criar engine
engine2 = create_engine(connection_string2)

# Executar a consulta e carregar os dados em um DataFrame
try:
    with engine2.connect() as conn:
        
        query3 = """SELECT DISTINCT customer_name FROM sales WHERE invoice_date >= '2022-05-01';"""

        customers_sales = pd.read_sql_query(query3, conn)
        
except Exception as e:
    print(f"Erro ao executar a consulta: {e}")

# Renomeando as colunas Customer_original
Customer_original = Customer_original.rename(columns = {'party_name':'Party Name',
                                    'description':'Description',
                                    'salespname':'Sales Person',
                                    'document_number':'Document Number',
                                    'organization_name':'Organization Name',
                                    'trx_number':'Trx Number',
                                    'trx_date':'Date'})

# Transformando para datetime a coluna de data Customer_original
Customer_original['Date'] = pd.to_datetime(Customer_original['Date'])


# Renomeando as colunas Supplier_original
Supplier_original = Supplier_original.rename(columns = {'organization_name':'Organization Name',
                                    'entity_name':'Remetente',
                                    'cnpj':'CNPJ Remetente',
                                    'invoice_type':'Invoice Type',
                                    'client_location':'Client Location',
                                    'cost_center':'Cost Center',
                                    'invoice_num':'Invoice Number',
                                    'invoice_date': 'Date'})


# Funcao para fazer o upload dos arquivos no Azure Blobs

az_storage_account_strin_conn = os.getenv("az_storage_account")
container_name = os.getenv("container_name")

def upload_to_blob(local_file_path,blob_name):
    
    time.sleep(10)

    try:
        blob_service_client = BlobServiceClient.from_connection_string(az_storage_account_strin_conn)

        container_client = blob_service_client.get_container_client(container_name)

        # Criar um cliente para o Blob especifico
        blob_client = container_client.get_blob_client(blob_name)

        with open(local_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        print(f"Upload concluido! Arquivo '{blob_name}' enviado para o Azure Blob Storage.")

    except Exception as e:
        print(f"Erro ao fazer upload: {str(e)}")

# Subindo o arquivo de contas
upload_to_blob(r"c:\Users\classificado_contas_contabeis.xlsx","classificado_contas_contabeis.xlsx")


# Tratando a tabela de Supplier 

Supplier_original = Supplier_original[Supplier_original['Invoice Type'].str.upper() != "TRANSFERENCIA PARA INDUSTRIALIZACAO DA MESMA EMPRESA"].copy()

Supplier_original['Date'] = pd.to_datetime(Supplier_original['Date'])

Supplier_original['Client Location'] = Supplier_original['Client Location'].astype('Int64')

Supplier_original['Cost Center'] = Supplier_original['Cost Center'].astype('Int64')

Supplier_original['Client Location'] = Supplier_original['Client Location'].astype(str)

Supplier_original['Cost Center'] = Supplier_original['Cost Center'].astype(str)


#Tratando a coluna 'Sales Person' de Customer e gerando um df com os valores unicos da coluna

Customer = Customer_original.copy()

Customer['Sales Person'] = Customer['Sales Person'].astype(str)

Customer['Sales Person'] =Customer['Sales Person'].str.upper()

Customer['Sales Person'] = Customer['Sales Person'].str.strip()

Customer['Sales Person'] = Customer['Sales Person'].apply(lambda x: unidecode(x) if isinstance(x, str) else x)

Customer_uniques = pd.DataFrame(Customer['Sales Person'].unique())

Customer_uniques.columns = ['Sales Person']


#Tratando a tabela CCs
#Funcao que encontra primeira palavra
def first_word(text):
    words = text.split()  # Dividindo a string em uma lista de palavras
    if len(words) >= 1:  # Verifica se há pelo menos uma palavra
        return words[0]  # Retorna a última palavra
    else:
        return ''  # Retorna uma string vazia se não há palavras

#Funcao que encontra última palavra
def last_word(text):
    words = text.split()  # Dividindo a string em uma lista de palavras
    if len(words) >= 1:  # Verifica se há pelo menos uma palavra
        return words[-1]  # Retorna a última palavra
    else:
        return ''  # Retorna uma string vazia se não há palavras

#Funcao todas as palavras menos a primeira
def all_but_first_word(text):
    words = text.split()  # Dividindo a string em uma lista de palavras
    if len(words) > 1:  # Se há mais de uma palavra
        return ' '.join(words[1:])  # Retorna todas as palavras após a primeira
    else:
        return ''  # Retorna uma string vazia se só tem uma palavra ou nenhuma
    
#Funcao para achar a melhor correspondencia
def get_best_match(row):
    # Obtem o melhor match em Customer_uniques['Sales Person'] para cada 'Sales Person CCs' em CCs 
    best_match = process.extractOne(row['Sales Person CCs'], Customer_uniques['Sales Person'].tolist())
    return best_match[0]  

#Extraindo o arquivo e tratando       
CCs = pd.read_excel(r"C:\Users\powerbi\Alocação Pessoas CC.xlsx")

CCs = CCs.iloc[:, [2, 3]]

CCs.columns = ['Sales Person CCs', 'CC']

#Tratando coluna 'CC'
CCs['CC'] = CCs['CC'].astype(str)
CCs['CC'] = CCs['CC'].str.replace(".0","")
CCs['CC'] = CCs['CC'].str.strip()

#Mantendo apenas o CC de vendas
CCs = CCs[CCs['CC'].str[0] == "6"]

#Tratando coluna 'Sales Person CCs'
CCs['Sales Person CCs'] = CCs['Sales Person CCs'].astype(str)

CCs['Sales Person CCs'] = CCs['Sales Person CCs'].str.upper()

CCs['Sales Person CCs'] = CCs['Sales Person CCs'].str.strip()

CCs['Sales Person CCs'] = CCs['Sales Person CCs'].apply(lambda x: unidecode(x) if isinstance(x, str) else x) #excluindo os acentos

CCs = CCs[['Sales Person CCs','Centro de Custo']]

#Encontrando a melhor corresondência em 
CCs['Best Match Customer'] = CCs.apply(get_best_match,axis=1)

CCs['1 Verificação'] = np.where(CCs['Sales Person CCs'].apply(first_word) != CCs['Best Match Customer'].apply(first_word),"Excluir","")

CCs = CCs[CCs['1 Verificação'] != "Excluir"]

CCs['Sobrenome Sales Person CCs'] = CCs['Sales Person CCs'].apply(all_but_first_word)

CCs['Último nome Customer'] = CCs['Best Match Customer'].apply(last_word)

CCs['2 Verificação'] = CCs.apply(lambda row: "Ok" if isinstance(row['Último nome Customer'], str) and isinstance(row['Sobrenome Sales Person CCs'], str) and row['Último nome Customer'] in row['Sobrenome Sales Person CCs'] else "", axis=1)

CCs = CCs[CCs['2 Verificação'] == "Ok"]

CCs = CCs[['Sales Person CCs','Centro de Custo','Best Match Customer']]


#Gerando a coluna Centro de Custo da tabela Customer

CCs = CCs.drop_duplicates(subset='Best Match Customer')

#Primeiro merge
Customer = pd.merge(Customer,CCs,how = "left", left_on='Sales Person',right_on = 'Best Match Customer')

Customer.rename(columns = {'Centro de Custo': 'Centro de Custo Vendedores'},inplace=True)

# Segundo Merge
Customer = pd.merge(Customer,CCs,how = "left", left_on='Party Name',right_on = 'Best Match Customer')

# Condicional para puxar o CC
Customer['Centro de Custo Vendedores'] = np.where(Customer['Centro de Custo Vendedores'].isnull(), Customer['Centro de Custo'], Customer['Centro de Custo Vendedores'])

# Droping unnecssary columns
Customer = Customer.drop(columns=['Sales Person CCs_x','Best Match Customer_x','Sales Person CCs_y','Centro de Custo','Best Match Customer_y'])

Customer.drop(columns=['Centro de Custo Vendedores'],inplace=True)

Customer['Centro de Custo'] = np.where(pd.isnull(Customer['Centro de Custo']),"-",Customer['Centro de Custo'])


#Gerando a coluna Item da tabela Customer

Customer['Item'].replace(to_replace={'nan':"550015-000000"},inplace=True)

Customer['Item'].fillna(value="550015-000000", inplace=True)


#Gerando a coluna CNPJ Destinatário na tabela Customer

#Função para ajustar cpf
def ajustar_cpf (cpf):
    return cpf.zfill(11)

#Função de formatação
def formatar_documento(documento):
    documento = ''.join(filter(str.isdigit, documento))

    if len(documento) == 14:  # CNPJ
        return f"{documento[:2]}.{documento[2:5]}.{documento[5:8]}/{documento[8:12]}-{documento[12:]}"
    elif len(documento) == 11:  # CPF
        return f"{documento[:3]}.{documento[3:6]}.{documento[6:9]}-{documento[9:]}"
    else:
        return "Formato inválido"  # Trata casos onde o documento não tem 11 nem 14 dígitos

Customer['Document Number'] = Customer['Document Number'].astype(str)

Customer['Caracteres'] = Customer['Document Number'].str.len()

Customer['CNPJ Destinatário'] = np.where(Customer['Caracteres'] == 10, Customer['Document Number'].apply(ajustar_cpf),Customer['Document Number'])

Customer['CNPJ Destinatário'] = Customer['CNPJ Destinatário'].apply(formatar_documento)

Customer = Customer[Customer['CNPJ Destinatário'] != "Formato inválido"]


#Gerando a coluna CNPJ Remetente para Customer

Customer['Destinatário'] = Customer['Party Name']

Customer['CNPJ Destinatário'] = Customer['CNPJ Destinatário'].str[:18]

Customer['Trx Number'] = Customer['Trx Number'].astype(str)

Customer.rename(columns = {'Description': 'Invoice Type','Trx Number': 'Invoice Number'},inplace=True)

Customer = Customer[['Remetente','CNPJ Remetente','Destinatário','CNPJ Destinatário','Centro de Custo','Item','Invoice Type','Invoice Number','Date']]


# Gerando o dataframe retorno_embalagens

# Copiando o dataframe Customer
retorno_embalagens = Customer.copy()

# Filtrando apenas as linhas que possuem clientes em destinatários pois esse tipo de nota nao sao integradas ao oracle
retorno_embalagens = retorno_embalagens[retorno_embalagens['Destinatário'].str.upper().isin(customers_sales['customer_name'])]

# Filtrando apenas as linhas que possuem centro de custo
retorno_embalagens = retorno_embalagens[retorno_embalagens['Centro de Custo'] != "-"]

# Invertendo a coluna de destinatário para remetente
retorno_embalagens['Remetente2'] = retorno_embalagens['Destinatário']
retorno_embalagens['CNPJ Remetente2'] = retorno_embalagens['CNPJ Destinatário']

# Invertendo a coluna de remetente para destinatário
retorno_embalagens['Destinatário2'] = retorno_embalagens['Remetente']
retorno_embalagens['CNPJ Destinatário2'] = retorno_embalagens['CNPJ Remetente']

# Dropando as colunas originais de destinatário e remetente
retorno_embalagens = retorno_embalagens.drop(columns=['Destinatário', 'CNPJ Destinatário', 'Remetente', 'CNPJ Remetente'])

# Renomeando as colunas novas
retorno_embalagens = retorno_embalagens.rename(columns={'Remetente2':'Remetente', 
                                                        'CNPJ Remetente2': 'CNPJ Remetente',
                                                        'Destinatário2': 'Destinatário',
                                                        'CNPJ Destinatário2': 'CNPJ Destinatário'})

# Ajustando a coluna 'Item'
retorno_embalagens['Item'] = "550398-000000"


#Gerando a coluna Centro de Custo da tabela Supplier

Supplier = Supplier_original.copy() 

Supplier['Centro de Custo'] = "0000." + Supplier['Client Location'] + "." + Supplier['Cost Center']

Supplier.loc[Supplier['Centro de Custo'] == "0000.0.0",'Centro de Custo'] = '-'

Supplier['Centro de Custo'] = np.where(Supplier['Centro de Custo'].str[4:7] == ".0.","0000"+"."+"0000"+"."+Supplier['Cost Center'],Supplier['Centro de Custo'])

Supplier['CNPJ Remetente'] = Supplier['CNPJ Remetente'].str.replace('[\s\u00A0]+', '', regex=True)

Supplier['CNPJ Remetente'] = Supplier['CNPJ Remetente'].str[1:]

Supplier['CNPJ Remetente'] =Supplier['CNPJ Remetente'].astype(str)

Supplier['CNPJ Remetente'] = Supplier['CNPJ Remetente'].apply(formatar_documento)

Supplier = Supplier[Supplier['CNPJ Remetente']!="Formato inválido"]


#Gerando as colunas CNPJ Remetente e CNPJ Destinatário na tabela Supplier

Supplier['Invoice Number'] = Supplier['Invoice Number'].astype(str)

Supplier['Invoice Number'] = np.where(Supplier['Invoice Number'].str[-3] == ".", Supplier['Invoice Number'].str[:-3],Supplier['Invoice Number'])

Supplier = Supplier[['CNPJ Remetente','Remetente','Destinatário','CNPJ Destinatário','Centro de Custo','Item','Invoice Type','Invoice Number','Date']]


#Concatando os dois dataframes

CTEs_database = pd.concat([Customer, Supplier], ignore_index=True)

def clean_df(df):

    df['Item'] = df['Item'].fillna("-")

    df['Item'].replace(to_replace={'nan' : "-" },inplace=True)

    df = df[df['CNPJ Destinatário']!= "..-"]

    df = df[df['CNPJ Remetente']!= "..-"]

    df.dropna(subset=['CNPJ Remetente'], inplace=True)

    df.dropna(subset=['CNPJ Destinatário'], inplace=True)

    df = df[df['CNPJ Remetente'] != "nan"]

    df = df[df['CNPJ Destinatário'] != "nan"]

    df = df[['Remetente','CNPJ Remetente','Destinatário','CNPJ Destinatário','Centro de Custo','Item','Invoice Type','Invoice Number','Date']]
    
    return df

# Aplicando a funcao de limpeza
CTEs_database = clean_df(CTEs_database)

#Conexão ao banco
nome_da_tabela = 'Invoices'

# String de conexão
connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}'

# Criar engine
engine = create_engine(connection_string)

# Deletando todos os dados da tabela antes de inserir novos
with engine.begin() as conn:
    conn.execute(text(f"DELETE FROM {nome_da_tabela}"))  # Modificação aqui para usar text

# Supondo que você tenha um DataFrame chamado CTEs_database pronto para ser inserido
CTEs_database.to_sql(nome_da_tabela, con=engine, if_exists='append', index=False)


#Configurando a segunda conexão ao banco

#Concatando os tres dataframes
CTEs_database = pd.concat([Customer, Supplier, retorno_embalagens], ignore_index=True)

# Aplicando a funcao de limpeza
CTEs_database = clean_df(CTEs_database)

# Tabela para salvar a base de treino e teste no SQL Server
nome_da_tabela = 'base_rna'

# String de conexão
connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}'

# Criar engine
engine = create_engine(connection_string)

# Deletando todos os dados da tabela antes de inserir novos
with engine.begin() as conn:
    conn.execute(text(f"DELETE FROM {nome_da_tabela}"))

# Salvando o df no SQL Server
CTEs_database[['Remetente', 'CNPJ Remetente', 'Destinatário', 'CNPJ Destinatário', 'Centro de Custo', 'Item']].to_sql(nome_da_tabela, con=engine, if_exists='append', index=False)

#Verificando o hardware e definindo as seeds

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

print(device)

#Inicialização
seed = 42

#Pytorch
torch.manual_seed(seed) 

#Numpy
np.random.seed(seed)

#Funções nativas do Python
random.seed(seed)


#Visualizando o dataset

#Gerando uma cópia do original
df = CTEs_database.copy()

#Mantendo as colunas necessarias
df = df[['CNPJ Remetente','CNPJ Destinatário','Centro de Custo','Item']]

#Transformando para string
df[['CNPJ Remetente','CNPJ Destinatário','Centro de Custo','Item']] = df[['CNPJ Remetente','CNPJ Destinatário','Centro de Custo','Item']].astype(str)

entrada = df[['CNPJ Remetente','CNPJ Destinatário']].copy()

entrada.to_excel(entrada_path)

upload_to_blob(entrada_path,"entradaRNA2.xlsx")

#Dummizando as colunas X e criando os labels Y

# Dummizando as colunas X
dummies_x = pd.get_dummies(df[['CNPJ Remetente','CNPJ Destinatário']])

# Criando o objeto LabelEncoder para as colunas Y
label_encoder_cc = LabelEncoder() # Centro de Custo
label_encoder_item = LabelEncoder() # Item

# Tratando as colunas Y para labels
df['Centro de Custo'] = label_encoder_cc.fit_transform(df['Centro de Custo'])
df['Item'] = label_encoder_item.fit_transform(df['Item'])

# Criando DataFrames para mapear índices de classe para as classes originais de cada coluna Y
mapeamento_cc = pd.DataFrame({
    'Classe Original CC': label_encoder_cc.classes_,
    'Índice CC': range(len(label_encoder_cc.classes_))
})

mapeamento_item = pd.DataFrame({
    'Classe Original Item': label_encoder_item.classes_,
    'Índice Item': range(len(label_encoder_item.classes_))
})

#Transformando para array e salvando os mapeamentos em arquivos .xlsx
mapeamento_cc_path = r"C:\Users\RNA 2 VM (crítico)\mapeamento_ccVM.xlsx"
mapeamento_cc.to_excel(mapeamento_cc_path,index=False)
upload_to_blob(mapeamento_cc_path,"mapeamento_ccVM.xlsx")
mapeamento_item_path = r"C:\Users\RNA 2 VM (crítico)\mapeamento_itemVM.xlsx"
mapeamento_item.to_excel(mapeamento_item_path,index=False)
upload_to_blob(mapeamento_item_path,"mapeamento_itemVM.xlsx")

#Concatenando o dataframe dummies_x com os labels Y
df_tratado = pd.concat([dummies_x, df[['Centro de Custo','Item']]], axis=1)


#Separando os conjuntos X e Y e contando o número de dimensões

# Conjunto X
df_x = dummies_x

# Conjunto Y
df_y = pd.DataFrame(df[['Centro de Custo','Item']])

#X
num_categorias_x = len(df_x.columns)
print("Número de categorias X:", num_categorias_x)

#Y cc
num_categorias_y_cc = df_y['Centro de Custo'].nunique()
print("Número de categorias Y cc:", num_categorias_y_cc)

#Y Item
num_categorias_y_item = df_y['Item'].nunique()
print("Número de categorias Y Item:", num_categorias_y_item)

num_categorias = {"num_categorias_x":[num_categorias_x],"num_categorias_y_cc":[num_categorias_y_cc],"num_categorias_y_item":[num_categorias_y_item]}

num_categorias = pd.DataFrame(num_categorias)

num_categorias_path = r"C:\Users\num_categoriasVM.xlsx"

num_categorias.to_excel(num_categorias_path,index=False)

upload_to_blob(num_categorias_path,"num_categoriasVM.xlsx")


#Convertendo para array e depois para tensor

centro_custo = torch.LongTensor(df_y['Centro de Custo'].to_numpy())
item = torch.LongTensor(df_y['Item'].to_numpy())

# Agora, X e Y são preparados como tensores
X = torch.FloatTensor(df_x.to_numpy())

# TensorDataset agora recebe uma tupla como segundo argumento para Y
dataset = TensorDataset(X, centro_custo, item)


# Definindo o tamanho dos conjuntos de treino e teste

tamanho_treino = int(0.75 * len(dataset))  # 75% para treino
tamanho_teste = len(dataset) - tamanho_treino  # o restante para teste

# O restante do código para divisão e criação de DataLoaders permanece o mesmo
treino_set, teste_set = random_split(dataset, [tamanho_treino, tamanho_teste])
treino_loader = DataLoader(treino_set, batch_size=300, shuffle=True)
teste_loader = DataLoader(teste_set, batch_size=300, shuffle=False)


#Instaciando a rede

torch.manual_seed(42)

class RedeComMultiplasSaidas(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes_cc, num_classes_item):
        super(RedeComMultiplasSaidas, self).__init__()
        # A parte comum da rede
        self.sequential_part = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        # Cabeças separadas para as saídas
        self.centro_custo_head = nn.Linear(hidden_size // 2, num_classes_cc)
        self.item_head = nn.Linear(hidden_size // 2, num_classes_item)
       
    def forward(self, x):
        x = self.sequential_part(x)
        centro_custo_out = self.centro_custo_head(x)
        item_out = self.item_head(x)
        return centro_custo_out, item_out


input_size  = num_categorias_x #número de dimensões de X
hidden_size = 500

net = RedeComMultiplasSaidas(input_size, hidden_size, num_categorias_y_cc, num_categorias_y_item)
net = net.to(device)


# Função de Perda
criterion = nn.CrossEntropyLoss().to(device)

# Otimizador: Descida do Gradiente
# Stochastic Gradient Descent
optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=5e-3)


#Treinando e testando a rede

# Configuração para o loop de treinamento e teste
num_epochs = 50

accuracies_treino_cc = []
accuracies_treino_item = []
accuracies_teste_cc = []
accuracies_teste_item = []


for epoch in range(num_epochs):
    # Treinamento
    net.train()
    total_samples = 0
    correct_predictions_cc = 0
    correct_predictions_item = 0

    for X_batch, centro_custo_batch, item_batch in treino_loader:
        X_batch = X_batch.to(device)
        centro_custo_batch = centro_custo_batch.to(device)
        item_batch = item_batch.to(device)

        optimizer.zero_grad()

        # Forward
        centro_custo_out, item_out = net(X_batch)

        # Calcula a perda para cada saída
        loss_centro_custo = criterion(centro_custo_out, centro_custo_batch)
        loss_item = criterion(item_out, item_batch)
        loss_total = loss_centro_custo + loss_item

        # Backward + Otimização
        loss_total.backward()
        optimizer.step()
        
        # Calcula acurácias
        _, predicted_cc = torch.max(centro_custo_out.data, 1)
        _, predicted_item = torch.max(item_out.data, 1)
        total_samples += centro_custo_batch.size(0)
        correct_predictions_cc += (predicted_cc == centro_custo_batch).sum().item()
        correct_predictions_item += (predicted_item == item_batch).sum().item()

    acc_cc = 100 * correct_predictions_cc / total_samples
    acc_item = 100 * correct_predictions_item / total_samples
    accuracies_treino_cc.append(acc_cc)
    accuracies_treino_item.append(acc_item)

    print(f'Epoch {epoch+1}/{num_epochs}, Treino - Acurácia Centro de Custo: {acc_cc:.2f}%, Acurácia Item: {acc_item:.2f}%')

    # Teste
    net.eval()  # Prepara o modelo para avaliação
    total_samples_test = 0
    correct_predictions_cc_test = 0
    correct_predictions_item_test = 0
    with torch.no_grad():
        for X_batch, centro_custo_batch, item_batch in teste_loader:
            X_batch = X_batch.to(device)
            centro_custo_batch = centro_custo_batch.to(device)
            item_batch = item_batch.to(device)

            # Forward
            centro_custo_out, item_out = net(X_batch)
            
            # Calcula acurácias
            _, predicted_cc_test = torch.max(centro_custo_out.data, 1)
            _, predicted_item_test = torch.max(item_out.data, 1)
            total_samples_test += centro_custo_batch.size(0)
            correct_predictions_cc_test += (predicted_cc_test == centro_custo_batch).sum().item()
            correct_predictions_item_test += (predicted_item_test == item_batch).sum().item()

    acc_cc_test = 100 * correct_predictions_cc_test / total_samples_test
    acc_item_test = 100 * correct_predictions_item_test / total_samples_test
    accuracies_teste_cc.append(acc_cc_test)
    accuracies_teste_item.append(acc_item_test)

    print(f'Epoch {epoch+1}/{num_epochs}, Teste - Acurácia Centro de Custo: {acc_cc_test:.2f}%, Acurácia Item: {acc_item_test:.2f}%')

#Salvando o modelo treinado
torch.save(net.state_dict(), 'modelo_treinado_RNA2VM.pth')
upload_to_blob(r"c:\Users\modelo_treinado_RNA2VM.pth", "modelo_treinado_RNA2VM.pth")


# Converta listas ou arrays em DataFrames pandas
accuracies_treino_item_df = pd.DataFrame(accuracies_treino_item, columns=['Train Item'])
accuracies_teste_item_df = pd.DataFrame(accuracies_teste_item, columns=['Test Item'])
accuracies_treino_cc_df = pd.DataFrame(accuracies_treino_cc, columns=['Train Cost Center'])
accuracies_teste_cc_df = pd.DataFrame(accuracies_teste_cc, columns=['Test Cost Center'])

Accuracies = pd.concat([
    accuracies_treino_item_df,
    accuracies_teste_item_df,
    accuracies_treino_cc_df,
    accuracies_teste_cc_df
], axis=1)

Accuracies.rename(columns={
    Accuracies.columns[0]: 'Train Item',
    Accuracies.columns[1]: 'Test Item',
    Accuracies.columns[2]: 'Train Cost Center',
    Accuracies.columns[3]: 'Test Cost Center'
}, inplace=True)

Accuracies['Train Item'] = Accuracies['Train Item'] / 100

Accuracies['Test Item'] = Accuracies['Test Item']/100

Accuracies['Train Cost Center'] = Accuracies['Train Cost Center']/100

Accuracies['Test Cost Center'] = Accuracies['Test Cost Center']/100

# Calculate the 'Epoch' for each row and increment by 1
Accuracies['Epoch'] = range(1, len(Accuracies) + 1)

# Assign the current date and time to all rows in the 'Date' column
Accuracies['Date'] = datetime.now()

# Inserindo os dados no DataFrame para a tabela SQL
Accuracies.to_sql('Accuracies', con=engine, if_exists='append', index=False)
