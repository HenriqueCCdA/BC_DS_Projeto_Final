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
    
def variaveis_explicativas(dados):
    '''
    --------------------------------------------------------
    Mostra o numero de linhas e colunas do DataFrame
    --------------------------------------------------------
    @param dados - dataFrame
    --------------------------------------------------------
    @return retorna um lista com os nomes das colunas
    --------------------------------------------------------
    '''
    
    for i, colunas in enumerate(dados.columns[:-1]):
        print(f'var[{i:3d}] -> {colunas}')
        
        
    return dados.columns[:-1]