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

try:
    from src.ml import treina_modelo_grid_search_cv, treina, cv_val_split, desempenho_dos_modelos 
    from src.info import proporcao_y, numero_teste_treino_val, dimensao_dados, variaveis_explicativas
except:
    from ml import treina_modelo_grid_search_cv, treina, cv_val_split, desempenho_dos_modelos 
    from info import proporcao_y, numero_teste_treino_val, dimensao_dados, variaveis_explicativas
    
def treinamentos(path='dados_featurewiz.csv', n_iter=10, n_splits=5, n_repeats=10, verbose=0):  
    
    if (verbose):
        print(f'Versoes das biblioteca')
        print(f'scipy      : {sc.__version__}')
        print(f'sklearn    : {sk.__version__}')
        print(f'pandas     : {pd.__version__}')
        print(f'numpy      : {np.__version__}')
        print(f'matplotlib : {mpl.__version__}')
   
    seed     = 1471523

    dados = pd.read_csv(path)
    
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
    parameters = {'strategy' : ['stratified', 'most_frequent', 'prior', 'uniform']}

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
    parameters = {'C': uniform(loc=0, scale=4)}
    modelo = LogisticRegression(max_iter=1000, tol=1e-6)
    melhor_modelo_lr, res = treina(modelo = modelo,
                        x = x_cv, 
                        y = y_cv, 
                        parameters = parameters, 
                        n_splits = n_splits, 
                        n_repeats = n_repeats, 
                        n_iter = n_iter, 
                        plota = False,
                        n = 10,
                        rng=rng)
    
    # Arvore de decisão
    print('Treinamento DecisionTreeClassifier')
    rng = RandomState(MT19937(SeedSequence(seed)))
    parameters = {'max_depth'       : np.arange(1, 21),
              'criterion'       : ['gini', 'entropy'],
              'min_samples_leaf': np.arange(1, 6),
              'max_leaf_nodes'  : np.arange(2, 6)
             }
    modelo = DecisionTreeClassifier()
    melhor_modelo_arvore, res = treina(modelo = modelo,
                            x = x_cv, 
                            y = y_cv, 
                            parameters = parameters, 
                            n_splits = n_splits, 
                            n_repeats = n_repeats, 
                            n_iter = n_iter, 
                            plota = False,
                            n = 10,
                            rng=rng)
    # RandomForestClassifier
    print('Treinamento RandomForestClassifier')
    rng = RandomState(MT19937(SeedSequence(seed)))
    parameters = {'n_estimators'    : [10, 50, 100, 150, 200, 250, 300],
                  'max_depth'       : np.arange(1, 21),
                  'criterion'       : ['gini', 'entropy'],
                  'min_samples_leaf': np.arange(1, 6),
                  'max_leaf_nodes'  : np.arange(2, 6)
                 }
    modelo = RandomForestClassifier(random_state=rng)
    melhor_modelo_forest, res = treina(modelo = modelo,
                              x = x_cv, 
                              y = y_cv, 
                              parameters = parameters, 
                              n_splits = n_splits, 
                              n_repeats = n_repeats, 
                              n_iter = n_iter, 
                              plota = False,
                              n = 10,
                              rng=rng)
    # SVC
    print('Treinamento SVC')
    rng = RandomState(MT19937(SeedSequence(seed)))
    parameters = {'kernel'          : ['linear', 'poly', 'rbf', 'sigmoid'],
              'C'               : uniform(loc=0, scale=2),
              'gamma'           : ['scale', 'auto'],
              'shrinking'       : [True, False]
             }

    modelo = SVC(probability=True)
    melhor_modelo_svc, res = treina(modelo = modelo,
                              x = x_cv, 
                              y = y_cv, 
                              parameters = parameters, 
                              n_splits = n_splits, 
                              n_repeats = n_repeats, 
                              n_iter = n_iter, 
                              plota = False,
                              n = 10,
                              rng=rng)
    
    # KNeighbors
    print('Treinamento KNeighbors')
    rng = RandomState(MT19937(SeedSequence(seed)))
    parameters = {'n_neighbors'     : np.arange(1, 11),
              'p'               : [1, 2],
              'weights'         : ['uniform', 'distance'],
              'algorithm'       : ['auto', 'ball_tree', 'kd_tree', 'brute']
             }

    modelo = KNeighborsClassifier()

    melhor_modelo_kn, res = treina(modelo = modelo,
                                  x = x_cv, 
                                  y = y_cv, 
                                  parameters = parameters, 
                                  n_splits = n_splits, 
                                  n_repeats = n_repeats, 
                                  n_iter = n_iter, 
                                  plota = False,
                                  n = 10,
                                  rng=rng)
    
    modelos = [melhor_modelo_dummy, 
               melhor_modelo_lr, 
               melhor_modelo_arvore, 
               melhor_modelo_forest, 
               melhor_modelo_svc,
               melhor_modelo_kn] 
    
    df = desempenho_dos_modelos(modelos, x_val, y_val)
    if (verbose):
        print('Resultados:')
        print(df)  
    return df
    
import sys
def main(argv):        
   
    path, n_iter, n_splits, n_repeats = argv[1], int(argv[2]), int(argv[3]), int(argv[4])
    
    treinamentos(path=path, n_iter=n_iter, n_splits=n_splits, n_repeats=n_repeats)
    
    
if __name__ == '__main__':
    main(sys.argv)
    
