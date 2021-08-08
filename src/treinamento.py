#
import pandas as pd
#
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
#
import matplotlib.pyplot as plt
#
import numpy as np
#
from scipy.stats import uniform
#
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

import sklearn as sk
import scipy   as sc
import matplotlib as mpl

from typing import Tuple

try:
    from src.ml import treina_modelo_grid_search_cv, treina, cv_val_split, desempenho_dos_modelos, obtem_nome_modelo 
    from src.info import proporcao_y, numero_teste_treino_val, dimensao_dados, variaveis_explicativas
    import src.hyperparametros as hp
except:
    from ml import treina_modelo_grid_search_cv, treina, cv_val_split, desempenho_dos_modelos, obtem_nome_modelo 
    from info import proporcao_y, numero_teste_treino_val, dimensao_dados, variaveis_explicativas
    import hyperparametros as hp
    
def treinamentos(path: str='dados_featurewiz.csv', 
                 n_iter: int=10, 
                 n_splits:int =5, 
                 n_repeats:int =10, 
                 verbose:int =0, 
                 seed:int = 1471523)->Tuple[pd.DataFrame, pd.DataFrame]:  
    '''
    ------------------------------------------------------------------------
    Treina os 6 modelos
    ------------------------------------------------------------------------
    @param path - Cominho completa da base de dados
    @param n_splits        - numero de divisões da base de dados do cv
    @param n_repeats       - numero de repetições que n_div e feita pelo
                             RepeatedStratifiedKFold
    @param n_iter          - numero de iterações do RandomizedSearchCV
    @verbose               - Define a grau de verbosida da funcoes
    @param seed            - Semente do gerado de numero aleatorios
    -------------------------------------------------------------------------
    @return Retorna a Tupla (a, b)
        a - O DataFrame com os resultados do dataset de validacao
        b - O DataFrame com os resultados do Cross Validation
    -------------------------------------------------------------------------
    '''    
    
    if (verbose):
        print(f'Versoes das biblioteca')
        print(f'scipy      : {sc.__version__}')
        print(f'sklearn    : {sk.__version__}')
        print(f'pandas     : {pd.__version__}')
        print(f'numpy      : {np.__version__}')
        print(f'matplotlib : {mpl.__version__}')
   
    dados = pd.read_csv(path)
    
    melhores_metriacas_cv_por_modelo={}
    
    dimensao_dados(dados)
    if (verbose):
        _ = variaveis_explicativas(dados)
    
    rng = RandomState(MT19937(SeedSequence(seed)))
    x_cv, x_val, y_cv, y_val = cv_val_split(dados, p_val = .10, rng=rng)
    
    if (verbose):
        print('\nProporcoes')
        proporcao_y(dados['ICU'])
        proporcao_y(y_val)
        proporcao_y(y_cv )
    
    numero_teste_treino_val(dados['ICU'], y_val, y_cv)

    # DummyClassifier
    print('Treinamento DummyClassifier')
    rng = RandomState(MT19937(SeedSequence(seed)))
    modelo = DummyClassifier(random_state=rng)
    parameters = hp.parameters_dummy

    _, melhor_modelo_dummy, _ = treina_modelo_grid_search_cv(modelo,                                                                                            
                                                             x_cv, 
                                                             y_cv,
                                                             parameters,
                                                             n_splits=n_splits,
                                                             n_repeats=n_repeats,
                                                             rng=rng)
    # Regressão logistica
    print('Treinamento LogisticRegression')
    rng = RandomState(MT19937(SeedSequence(seed)))
    parameters = hp.parameters_logistic_regression
    modelo = LogisticRegression(max_iter=1000, tol=1e-6)
    melhor_modelo_lr, res, melhor_metrica = treina(modelo = modelo,
                        x = x_cv, 
                        y = y_cv, 
                        parameters = parameters, 
                        n_splits = n_splits, 
                        n_repeats = n_repeats, 
                        n_iter = n_iter, 
                        plota = False,
                        n = 10,
                        rng=rng)
    
    melhores_metriacas_cv_por_modelo[obtem_nome_modelo(modelo)] =  melhor_metrica
    
    # Arvore de decisão
    print('Treinamento DecisionTreeClassifier')
    rng = RandomState(MT19937(SeedSequence(seed)))
    parameters = hp.parameters_decision_tree
    modelo = DecisionTreeClassifier()
    melhor_modelo_arvore, res, melhor_metrica = treina(modelo = modelo,
                            x = x_cv, 
                            y = y_cv, 
                            parameters = parameters, 
                            n_splits = n_splits, 
                            n_repeats = n_repeats, 
                            n_iter = n_iter, 
                            plota = False,
                            n = 10,
                            rng=rng)
    melhores_metriacas_cv_por_modelo[obtem_nome_modelo(modelo)] =  melhor_metrica
    
    # RandomForestClassifier
    print('Treinamento RandomForestClassifier')
    rng = RandomState(MT19937(SeedSequence(seed)))
    parameters = hp.parameters_random_forest
    modelo = RandomForestClassifier(random_state=rng)
    melhor_modelo_forest, res, melhor_metrica = treina(modelo = modelo,
                              x = x_cv, 
                              y = y_cv, 
                              parameters = parameters, 
                              n_splits = n_splits, 
                              n_repeats = n_repeats, 
                              n_iter = n_iter, 
                              plota = False,
                              n = 10,
                              rng=rng)
    melhores_metriacas_cv_por_modelo[obtem_nome_modelo(modelo)] =  melhor_metrica
    
    # SVC
    print('Treinamento SVC')
    rng = RandomState(MT19937(SeedSequence(seed)))
    parameters = hp.parameters_svc
    modelo = SVC(probability=True)
    melhor_modelo_svc, res, melhor_metrica = treina(modelo = modelo,
                              x = x_cv, 
                              y = y_cv, 
                              parameters = parameters, 
                              n_splits = n_splits, 
                              n_repeats = n_repeats, 
                              n_iter = n_iter, 
                              plota = False,
                              n = 10,
                              rng=rng)
    melhores_metriacas_cv_por_modelo[obtem_nome_modelo(modelo)] =  melhor_metrica
    
    # KNeighbors
    print('Treinamento KNeighbors')
    rng = RandomState(MT19937(SeedSequence(seed)))
    parameters = hp.parameters_KNeighbors
    modelo = KNeighborsClassifier()

    melhor_modelo_kn, res, melhor_metrica = treina(modelo = modelo,
                                  x = x_cv, 
                                  y = y_cv, 
                                  parameters = parameters, 
                                  n_splits = n_splits, 
                                  n_repeats = n_repeats, 
                                  n_iter = n_iter, 
                                  plota = False,
                                  n = 10,
                                  rng=rng)
    melhores_metriacas_cv_por_modelo[obtem_nome_modelo(modelo)] =  melhor_metrica
    
    modelos = [melhor_modelo_dummy, 
               melhor_modelo_lr, 
               melhor_modelo_arvore, 
               melhor_modelo_forest, 
               melhor_modelo_svc,
               melhor_modelo_kn] 
    
    df_val = desempenho_dos_modelos(modelos, x_val, y_val)
    df_cv  = pd.DataFrame(melhores_metriacas_cv_por_modelo).T.sort_values('media', ascending=False)
    if (verbose):
        print('Resultados no DataSet de validacao:')
        print(df_val)
        print('Resultados Cross Validation:')
        print(df_cv)          
        
    return df_val, df_cv
    
import sys

def main(argv):        
   
    if(len(argv) == 1):
        print('Numero de argumentos insuficiente:')
        print('Ex de uso: returnamento.py path n_iter, n_splits, n_repeats')
    else:
        path, n_iter, n_splits, n_repeats = argv[1], int(argv[2]), int(argv[3]), int(argv[4])
        treinamentos(path=path, n_iter=n_iter, n_splits=n_splits, n_repeats=n_repeats)
    
    
if __name__ == '__main__':
    main(sys.argv)
    
