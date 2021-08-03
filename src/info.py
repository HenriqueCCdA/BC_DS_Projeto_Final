import csv 

def dimensao_dados(dados):
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
    
def variaveis_explicativas(dados, ncols=2):
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
                print(f'col[{kk:3d}] -> {dados.columns[kk]:30s}', end=' ')
            else:
                fexit = True
                break
        if(fexit):
            break
        print()
        i += ncols        
        
    return columns
    
def mostra_todas_as_colunas_com(dados, string):
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
            
def escreve_somente_as_colunas(dados, arquivo='colunas.csv'):
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
                        
def proporcao_y(y):
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
        
def numero_teste_treino_val(y, y_val, y_cv):
    '''
    ------------------------------------------------------------------------
    Mostra o numero de entradas em cada dataset
    ------------------------------------------------------------------------
    '''
    n = len(y)
    print(f'Número total de entradas                         : {len(y)}')
    print(f'Número total de entradas para validacao          : {len(y_val)}')
    print(f'Número total de entradas para o Cross Validation : {len(y_cv)}')
    
def resultados_treinamento(resultados, modelo, hyperparametros, n = 5):
    '''
    Mostra os resltados do modelo treinado
    '''
    print(f'melhores hyperparametros : {hyperparametros}')
    print(f'Melhor modelo            : {modelo}')
    return resultados.head(n=n)


