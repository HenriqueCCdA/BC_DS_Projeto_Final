import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
# meus
try:
    from src.plota_graficos import plota_treino_teste_auc
    from src.info import resultados_treinamento
except:
    from plota_graficos import plota_treino_teste_auc
    from info import resultados_treinamento

def retorna_x_y(dados):
    '''
    ---------------------------------------------------------
    Retorna o DataFrame divido entre x e y
    ---------------------------------------------------------    
    @param dados - DataFrame de dados
    ---------------------------------------------------------
    @return retorna a tupla (x,y)
            x - varaiveis explicativas
            y - label
    ---------------------------------------------------------        
    '''
    return dados.iloc[:,:-1], dados.iloc[:,-1]

def intervalo_de_confianca(media, std):
    '''
    ---------------------------------------------------------
    Calcula o intervalo de confiança para uma dada media e
    desvio padrao
    ---------------------------------------------------------    
    @param media - media
    @param std   - desvio padrao
    ---------------------------------------------------------
    @return retorna a tupla (media-2*std, media + 2*std)
    ---------------------------------------------------------        
    '''
    return media-2*std, media + 2*std

def logistic_regression_sumary(modelo):
    '''
    ---------------------------------------------------------
    Mostra os parametros da regressão logistica apos o treino
    ---------------------------------------------------------    
    @param modelo - modelo de refressão logistica treinado
    ---------------------------------------------------------    
    '''
    print(f'Coeficientes: {modelo.coef_.shape[1]}')
    print(modelo.coef_)
    
    print('Intercepto:')
    print(modelo.intercept_)
    
    print('Numero de iteracoes:')
    print(modelo.n_iter_)

def roda_modelo_direto(modelo, x, y, seed = 42258):
    '''
    ---------------------------------------------------------
    Roda um modelo ML diretamente. 
    ---------------------------------------------------------    
    @param x      - DataFrame com os dados para o cv 
    @param y      - DataFrame com os dados para o cv
    @param seed   - semente dos numeros aletorios usado no 
                    np.random.seed
    ---------------------------------------------------------    
    '''
    
    np.random.seed(seed)
    
    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, stratify=y, test_size=0.10)

    print(f'Tamanho teste : {len(y_teste) :4d}')
    print(f'Tamanho treino: {len(y_treino):4d}')

    modelo.fit(x_treino, y_treino) 

    prob_treino = modelo.predict_proba(x_treino)
    prob_teste  = modelo.predict_proba(x_teste)

    auc_treino = roc_auc_score(y_treino, prob_treino[:,1])
    auc_teste  = roc_auc_score(y_teste , prob_teste[:,1])

    print(f"AUC Teste     : {auc_teste :.2f}")
    print(f"AUC Treino    : {auc_treino:.2f}")

def roda_modelo_cv(modelo, x, y, n_div = 5,
                   n_rep = 5, seed= 42258):
    '''
    ------------------------------------------------------------
    Roda um modelo ML utilizando o croosvalidation com  
    RepeatedStratifiedKFold
    ------------------------------------------------------------    
    @param modelo - modelo istânciado não treinado
    @param x      - DataFrame com os dados para o cv 
    @param y      - DataFrame com os dados para o cv
    @param n_div  - numero de divisões da base de dados do cv
    @param n_rep  - numero de repetições que n_div e feita pelo
                    RepeatedStratifiedKFold
    @param seed   - semente dos numeros aletorios usado no 
                    np.random.seed
    -------------------------------------------------------------
    @return retorna uma tupla (auc_medio_teste, auc_medio_treino)
    --------------------------------------------------------------
    '''  
    np.random.seed(seed)
  
    cv = RepeatedStratifiedKFold(n_splits = n_div, n_repeats = n_rep)
      
    resultados = cross_validate(modelo, x, y, cv= cv, 
                                scoring='roc_auc',
                                return_train_score=True)
  
    auc_medio_teste  = np.mean(resultados['test_score'])
    auc_medio_treino = np.mean(resultados['train_score'])
  
    auc_std_teste  = np.std(resultados['test_score'])
    auc_std_treino = np.std(resultados['train_score'])

    teste_int = intervalo_de_confianca(auc_medio_teste , auc_std_teste)
    treino_int = intervalo_de_confianca(auc_medio_treino, auc_std_treino)
    
    print(f'AUC Teste : {auc_medio_teste :.2f} - [{teste_int[0] :.2f} - {teste_int[1] :.2f}]')
    print(f'AUC Treino: {auc_medio_treino:.2f} - [{treino_int[0]:.2f} - {treino_int[1]:.2f}]')
    
    return auc_medio_teste, auc_medio_treino

def obtem_os_resultados_SearchCV(clf):
    '''
    ------------------------------------------------------------
    Obtem as principais informas do GridSearchCV ou 
    RandomizedSearchCV
    ------------------------------------------------------------    
    @param clf    - GridSearchCV ou RandomizedSearchCV
    -------------------------------------------------------------
    @return retorna uma tupla (a, b, c)
            a - DataFrame com as princiais informações
            b - O melhor modelo já retreinado com toda a base de
            dados (refit = True)
            c - Os melhores parametros
    --------------------------------------------------------------
    '''  
    aux = {'paramentros'    : clf.cv_results_['params'],
           'media_teste'    : clf.cv_results_['mean_test_score'],
           'media_treino'   : clf.cv_results_['mean_train_score'],
           'std_teste'      : clf.cv_results_['std_test_score'],
           'std_treino'     : clf.cv_results_['std_train_score'],
           'mean_fit_time'  : clf.cv_results_['mean_fit_time'],
           'std_fit_time'   : clf.cv_results_['std_fit_time'],
           'mean_score_time': clf.cv_results_['mean_score_time'],
           'std_score_time' : clf.cv_results_['std_score_time'],
           'rank_test_score': clf.cv_results_['rank_test_score']
          }       
    
    return pd.DataFrame(aux), clf.best_estimator_, clf.best_params_

def treina_modelo_randomized_search_cv(modelo,
                                       x,
                                        y,
                                        parameters,
                                        n_splits = 5,
                                        n_repeats = 5,
                                        n_iter = 10,
                                        seed = 14715):

    '''
    ------------------------------------------------------------
    Roda um modelo ML utilizando o Cross Validation com  
    RepeatedStratifiedKFold
    ------------------------------------------------------------    
    @param modelo      - modelo istânciado não treinado
    @param x           - DataFrame com os dados para o cv 
    @param y           - DataFrame com os dados para o cv
    @param paramentros - Dicionario com os parametros a serem testados
    @param n_splits    - numero de divisões da base de dados do cv
    @param n_repeats   - numero de repetições que n_div e feita pelo
                         RepeatedStratifiedKFold
    @param n_iter      - numero de iterações do RandomizedSearchCV
    @param seed        - semente dos numeros aletorios usado no 
                       np.random.seed
    -------------------------------------------------------------
    @return retorna uma tupla (a, b, c)
            a - DataFrame com as princiais informações
            b - O melhor modelo já retreinado com toda a base de
            dados (refit = True)
            c - Os melhores parametros
    --------------------------------------------------------------
    '''  
    
    
    np.random.seed(seed)

    cv = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats = n_repeats)

    clf = RandomizedSearchCV(modelo, parameters,
                             cv=cv, 
                             verbose=1,
                             scoring='roc_auc',
                             return_train_score=True,
                             n_iter=n_iter,
                             refit=True) 

    clf.fit(x, y)
    resultados, melhor_modelo, melhores_hyperparamentros = obtem_os_resultados_SearchCV(clf)

    return resultados, melhor_modelo, melhores_hyperparamentros

def treina_modelo_grid_search_cv(modelo,
                                 x,
                                 y,
                                 parameters,
                                 n_splits = 5,
                                 n_repeats = 5,
                                 n_iter = 10,
                                 seed = 14715):

    '''
    ------------------------------------------------------------
    Roda um modelo ML utilizando o Cross Validation com  
    RepeatedStratifiedKFold
    ------------------------------------------------------------    
    @param modelo      - modelo istânciado não treinado
    @param x           - DataFrame com os dados para o cv 
    @param y           - DataFrame com os dados para o cv
    @param paramentros - Dicionario com os parametros a serem testados
    @param n_splits    - numero de divisões da base de dados do cv
    @param n_repeats   - numero de repetições que n_div e feita pelo
                         RepeatedStratifiedKFold
    @param n_iter      - numero de iterações do RandomizedSearchCV
    @param seed        - semente dos numeros aletorios usado no 
                       np.random.seed
    -------------------------------------------------------------
    @return retorna uma tupla (a, b, c)
            a - DataFrame com as princiais informações
            b - O melhor modelo já retreinado com toda a base de
            dados (refit = True)
            c - Os melhores parametros
    --------------------------------------------------------------
    '''  
    
    
    np.random.seed(seed)

    cv = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats = n_repeats)

    clf =GridSearchCV(modelo, parameters,
                              cv=cv, 
                              verbose=1,
                              scoring='roc_auc',
                              return_train_score=True,
                              refit=True) 

    clf.fit(x, y)
    resultados, melhor_modelo, melhores_hyperparamentros = obtem_os_resultados_SearchCV(clf)

    return resultados, melhor_modelo, melhores_hyperparamentros

def cv_val_split(dados, p_val=0.1, seed = 14715):
    '''
    ----------------------------------------------------------------------------
    Divide os dados entre Cross Validation (treino+test) e validacao
    ----------------------------------------------------------------------------
    @param dados - dataFrame com os dados
    @param p_val - porcentagem dos dos que vão para validadação
    @param seed  - semente do gerador de numero aleatorios
    ----------------------------------------------------------------------------
    @return retorna a tupla (x_cv, x_val, y_cv, y_val)
            x_cv  - dados para o Cross Validation ( treino + teste)
            x_val - dados para a validacao
            y_cv  - dados para o Cross Validation ( treino + teste)
            y_val - dados para a validacao
    ----------------------------------------------------------------------------
    '''
    dados = dados.sample(frac=1).reset_index(drop=True)
    
    x, y = retorna_x_y(dados)

    x_cv, x_val, y_cv, y_val = train_test_split(x, y, stratify=y, random_state = seed
                                                , test_size=p_val)
    
    return x_cv, x_val, y_cv, y_val

def desempenho_dos_modelos(modelos, x, y):
    '''
    ---------------------------------------------------------------
    Plota a matriz de confusao 
    ---------------------------------------------------------------
    @param modelos  - lista com os modelos
    @param x        - x do modelo de ML
    @param y        - y do modelo de ML
    ---------------------------------------------------------------
    @return retona um dataFrame com as informação do desempenho
    ---------------------------------------------------------------
    ''' 

    tns, fps, fns, tps, aucs, names = [], [], [], [], [], []

    for modelo in modelos:
        name = type(modelo).__name__
        (tn, fp), (fn, tp) = confusion_matrix(y, modelo.predict(x))
        auc = roc_auc_score(y, modelo.predict_proba(x)[:, 1])
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)
        tps.append(tp)
        aucs.append(auc)
        names.append(name)

    modelos_desempenho = pd.DataFrame({'Name': names,
                                        'tn': tns,
                                        'fp': fps,
                                        'fn': fns,
                                        'tp': tps,
                                        'AUC': aucs
                                      }).sort_values('AUC', ascending=False, ignore_index=True)
    return modelos_desempenho

def treina(modelo, x, y, parameters, n_splits, n_repeats, n_iter, seed, titulo, n):
    '''
    ----------------------------------------------------------------------------
    Treina e mostra os resultados
    ----------------------------------------------------------------------------
    @param modelo      - modelo istânciado não treinado
    @param x           - DataFrame com os dados para o cv 
    @param y           - DataFrame com os dados para o cv
    @param paramentros - Dicionario com os parametros a serem testados
    @param n_splits    - numero de divisões da base de dados do cv
    @param n_repeats   - numero de repetições que n_div e feita pelo
                         RepeatedStratifiedKFold
    @param n_iter      - numero de iterações do RandomizedSearchCV
    @param seed        - semente dos numeros aletorios usado no 
                       np.random.seed
    @param titulo      - titulo do gradico
    @param n           - numero de linhas no DataFrame com os Res
    ------------------------------------------------------------------------------
    @return retorna uma tupla (a, b)
            a - O melhor modelo já retreinado com toda a base de
                dados (refit = True)
            b - DataFrame com os resultados
    ----------------------------------------------------------------------------
    '''

    resultados, melhor_modelo, hyperparametros  =\
        treina_modelo_randomized_search_cv(modelo,
                                          x,
                                          y,
                                          parameters,
                                          n_splits=n_splits,
                                          n_repeats=n_repeats,
                                          n_iter=n_iter,
                                          seed=seed)

    plota_treino_teste_auc(titulo, 
                         resultados['media_teste'],
                         resultados['media_treino'],
                         resultados['rank_test_score'],
                         hyperparametros)

    pd = resultados_treinamento(resultados, melhor_modelo, hyperparametros, n = n)

    return melhor_modelo, pd