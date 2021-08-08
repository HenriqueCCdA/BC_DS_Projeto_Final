import csv 
#
from typing import List, NewType, Dict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

Array1D = NewType('Arranjo 1D',  pd.Series)

def dimensao_dados(dados: pd.DataFrame):
    '''
    --------------------------------------------------------
    Mostra o numero de linhas e colunas do DataFrame
    --------------------------------------------------------
    @param dados - dataFrame
    --------------------------------------------------------
    '''
    nl, nc = dados.shape
    print(f"Numero de linhas : {nl} ")
    print(f"Numero de colunas: {nc} ")
    
def variaveis_explicativas(dados: pd.DataFrame, ncols: int =2)-> List[str]:
    '''
    --------------------------------------------------------
    Mostra o numero de linhas e colunas do DataFrame
    --------------------------------------------------------
    @param dados - dataFrame
    @param nclos - numero de colunas mostrada por linha
    --------------------------------------------------------
    @return retorna um lista com os nomes das colunas
    --------------------------------------------------------
    ''' 

    columns = dados.columns.drop('ICU')
    n = len(columns)
    i = 0
    fexit = False
    for k in range(0, n):
        print(end=' ')
        for j in range(0, ncols):
            kk = i+j
            if ( kk < n):
                print(f'col[{kk:3d}] -> {dados.columns[kk]:35s}', end=' ')
            else:
                fexit = True
                break
        if(fexit):
            break
        print()
        i += ncols        
        
    return columns
    
def mostra_todas_as_colunas_com(dados: pd.DataFrame, string: str):
    '''
    --------------------------------------------------------
    Mostra todas as colunas com a string no nome
    --------------------------------------------------------
    @param dados - dataFrame
    @param string - string que se quer procurar nos nomes 
                    das colunas
    --------------------------------------------------------
    ''' 
    for name in dados.columns:
        if string in name:
            print(name)
            
def escreve_somente_as_colunas(dados: pd.DataFrame, arquivo: str='colunas.csv'):
    '''
    ----------------------------------------------------------------------------
    Escreve o nome das colunas do dataframe em arquivo diferente
    ----------------------------------------------------------------------------
    @param dados   - DataFrame
    @param arquivo - Nome do arquivo de saida
    ----------------------------------------------------------------------------
    ''' 
    with open(arquivo, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(['numero', 'nome_colunas'])

        for i, row in enumerate(dados.columns):
            writer.writerow([i, row])
                        
def proporcao_y(y: pd.Series):
    '''
    ----------------------------------------------------------------------------
    Mostra a proporcao do y 
    ----------------------------------------------------------------------------
    @param y    - DataSeries
    ----------------------------------------------------------------------------
    ''' 
    p = y.value_counts(normalize=True)

    print(f"Proporcao do {y.name}")
    for l , v in zip(p.index, p.values):
        print(f"Campo {l} ->  {v*100:.2f}%")
        
def numero_teste_treino_val(y: Array1D, y_val: Array1D, y_cv: Array1D):
    '''
    ------------------------------------------------------------------------
    Mostra o numero de entradas em cada dataset
    ------------------------------------------------------------------------
    '''
    n = len(y)
    print(f'Número total de entradas                         : {len(y)}')
    print(f'Número total de entradas para validacao          : {len(y_val)}')
    print(f'Número total de entradas para o Cross Validation : {len(y_cv)}')
    
def resultados_treinamento(resultados: pd.DataFrame, 
                           modelo:BaseEstimator , 
                           hyperparametros: Dict, n: int = 5)->pd.DataFrame:
    '''
    --------------------------------------------------------------------------
    Mostra os resltados do modelo treinado
    --------------------------------------------------------------------------
    @param Resultados      - DataFrame com os resultados
    @param modelo          - Modelo de classifica do sklearn
    @param hyperparametros - Dicionario com os hyperparamentros
    @param n               - Numero linhas do DataFrame de resultados que serao 
                             retornados
    --------------------------------------------------------------------------
    '''
    print(f'melhores hyperparametros : {hyperparametros}')
    print(f'Melhor modelo            : {modelo}')
    return resultados.head(n=n)
