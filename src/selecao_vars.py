import numpy as np

def remove_coor_valor_de_corte(dados, valor_corte):
    '''
    ----------------------------------------------------------------------------
    Remove as colunas com valores de correlação maior que que o valor_corte
    ----------------------------------------------------------------------------
    @param dados        - DataFrame
    @param valor_corte -  Valor de corte que será usado para remoção das colunas
    ----------------------------------------------------------------------------
    @return Retorna um tupla (df, lista_string) 
            df - Novo dataFrame com as colunas removidades  
            lista_string - o nomes das colunas removidas
    ----------------------------------------------------------------------------
    '''  


    matrix_coor = dados.iloc[:,:-1].corr().abs()
    matrix_upper = matrix_coor.where(np.triu(np.ones(matrix_coor.shape).astype(bool), k = 1 ))
    excluir = [coluna for coluna in matrix_coor.columns if any(matrix_upper[coluna] > valor_corte)]

    return dados.drop(excluir, axis=1), excluir