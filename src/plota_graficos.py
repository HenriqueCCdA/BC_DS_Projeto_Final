import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve

#********************************************************************* 
def plot_barras(titulo, x, y, xlim, n_colors=10):
    '''
    ---------------------------------------------------------------
    Plota graficos de barras com seaborn 
    ---------------------------------------------------------------
    @param titulo   - titulo do grafico
    @param x        - valores eixo x
    @param y        - valores eixo y
    @param xlim     - limites no eixo x [x1, x2]
    @param n_colors - numero de cores na paleta de cores 
    ---------------------------------------------------------------
    '''  
    sns.set_style("white")

#    pal = sns.color_palette("tab10", 10)
    pal = sns.color_palette('BuPu', n_colors)

    fig, ax = plt.subplots(figsize=(10, 5))

    fig.text(x = 0.06, y = 0.95,
         s = titulo,
         fontsize=24, color = 'gray') 

    sns.barplot(y = y, 
            x = x*100, 
            ax = ax,
            palette = pal,
            orient = 'h')

# limite
    ax.set_xlim(xlim)

    ax.set_xlabel('Porcentagem (%)', loc = 'left', fontsize= 16, color='dimgray')
#ax.set_ylabel('Genêro', loc = 'top', fontsize= 16, color='dimgray')

    plt.xticks(fontsize= 16, color='gray')
    plt.yticks(fontsize= 16, color='gray')


    sns.despine()
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_color('gray')

    plt.show()
#*********************************************************************     
    
#*********************************************************************     
def plot_barras_grupos(dados, titulo):
    '''
    ---------------------------------------------------------------
    Plota graficos de barras com seaborn usando hue
    ---------------------------------------------------------------
    @param dados - dataframe dos dados
    @param titulo - titulo do grafico
    ---------------------------------------------------------------
    '''
    sns.set_style("white")
#    pal = sns.color_palette("tab10", 10)
    pal = sns.color_palette('BuPu', 2)

    fig, ax = plt.subplots(figsize=(10, 5))

    fig.text(x = 0.06, y = 0.95,
             s = titulo,
             fontsize=24, color = 'gray') 


    sns.barplot(data=dados, 
            y= 'grupo', 
            x = 'value', 
            hue = 'variable', 
            orient = 'h',
            palette=pal)

    # limite
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 6)

    ax.set_xlabel('Porcentagem (%)', loc = 'left', fontsize= 16, color='dimgray')
    ax.set_ylabel(None)

    plt.xticks(fontsize= 16, color='gray')
    plt.yticks(fontsize= 16, color='gray')


    sns.despine()
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_color('gray')

    ax.legend(fontsize=14, ncol=2)

    plt.show()
#********************************************************************* 

#********************************************************************* 
def plota_matriz_correlacao(dados, matriz = 'upper'):
    '''
    ----------------------------------------------------------------------------
    Plota a matiz de correlação
    ----------------------------------------------------------------------------
    @param dados  - DataFrame
    @param matriz - 'upper'    - triangula superior
                    'lower'    - triangular inferio
                    'complete' - a matriz completa
    ----------------------------------------------------------------------------
    '''

    matriz_coor = dados.iloc[:,:-1].corr().abs()
    
    if(matriz == 'upper' ):
        matriz_bool = np.triu(np.ones(matriz_coor.shape),  k = 1).astype(bool)
        matriz_coor = matriz_coor.where(matriz_bool, other=np.nan)
    elif(matriz == 'lower' ):
        matriz_bool = np.tril(np.ones(matriz_coor.shape),  k = -1).astype(bool)
        matriz_coor = matriz_coor.where(matriz_bool, other=np.nan)

    fig, ax = plt.subplots(figsize=(12,8))

    ax.set_title("Matriz de correlação", fontsize=24, loc='left', color = 'gray')

    myColors=['green', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray','blue' ]
    
    map = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
    
    
    ax = sns.heatmap(matriz_coor,
                     vmin=.0,
                     vmax=1.0,
                     center=0.5,
                     cmap=map,
                     xticklabels=[],
                     yticklabels=[],
                     linewidths=0.0,
                     ax=ax)
    
    colorbar = ax.collections[0].colorbar

    num = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5 , 0.6, 0.7, 0.8, 0.9, 1.0]
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks(num)
    colorbar.set_ticklabels(num)
    plt.show()    
#********************************************************************* 

#********************************************************************* 
def plota_treino_teste_auc(nome_modelo, teste_auc, treino_auc, rank, hp_melhor,
                           pasta_saida_fig='./',
                           f_save_fig=False):
    '''
    ------------------------------------------------------------
    Plota a curva AUC para o teste de hyperparamentros
    ------------------------------------------------------------    
    @param nome_modelo- Nome do modelo no titulo
    @param teste_auc  - Valores AUC dos Teste
    @param treino_auc - Valores AUC dos Treino
    @param rank       - colocação dos modelos
    @param hp_melhor  - hyperparametro que gera o melhor modelo
    @param f_save_fig - salva a figura em um arquivo
    @param pasta_saida_fig - diretorio onde as figuras serao salvas
    -------------------------------------------------------------
    ''' 
    
    fig, ax = plt.subplots(figsize=(16,8))

    fig.set_facecolor('white')
    ax.set_facecolor('white')
    
    for melhor_pos, v in enumerate(rank):
        if(v == 1):
            break
    
    fig.text(x = 0.08, y = 0.95,
         s = 'Seleção de Hyperparametros - ' + nome_modelo,
         fontsize=24, color = 'gray') 
    
    x = range(0, len(teste_auc))

    ax.plot(x,  teste_auc, color='green', lw=2)
    ax.plot(x, treino_auc, color='blue', lw=2)

    ax.set_xlim(x[0], x[-1]+1)
    ax.set_ylim(0.0, 1.01)

    ax.set_ylabel('AUC Médio', loc = 'top', fontsize= 16, color='dimgray')
    ax.set_xlabel('Número do teste do SearchCV', loc = 'left', fontsize= 16, color='dimgray')
    
    plt.xticks(x, fontsize= 14, color='gray')
    plt.yticks(fontsize= 14, color='gray')
    
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_color('gray')   
    
    ax.annotate(text='Treino', xy=(x[-1], max(treino_auc)), fontsize=16, color = 'blue')
    ax.annotate(text='Teste' , xy=(x[-1], max(teste_auc )), fontsize=16, color = 'green')
    
    ax.vlines(x=melhor_pos, ymin=0, ymax=2, ls='--', color='black')
    ax.annotate(text = 'Melhor conjuto:' + str(hp_melhor), xy=(melhor_pos+0.1, 0.2), rotation=0, fontsize=12)
        
    if (f_save_fig):
        plt.savefig( pasta_saida_fig + nome_modelo + "_treino_teste_auc.png")        
        
    plt.show()
       

#********************************************************************* 
    
#*********************************************************************     
def plota_matriz_de_confusao(modelos, x, y):
    '''
    ---------------------------------------------------------------
    Plota a matriz de confusao 
    ---------------------------------------------------------------
    @param modelos  - lista com os modelos
    @param x        - x do modelo de ML
    @param y        - y do modelo de ML
    ---------------------------------------------------------------
    '''    
    
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    ax = ax.reshape(-1)

    for i, modelo in enumerate(modelos):
        ax[i].set_title(type(modelo).__name__, fontsize=16)
        plot_confusion_matrix(modelo, x, y, ax=ax[i], normalize='all')
#*********************************************************************    

#*********************************************************************    
def plota_curva_roc(modelos, titulo, x, y):
    '''
    ---------------------------------------------------------------
    Plota a curva ROC 
    ---------------------------------------------------------------
    @param modelos  - lista com os modelos
    @param titulo   - titulo do grafico
    @param x        - x do modelo de ML
    @param y        - y do modelo de ML
    ---------------------------------------------------------------
    '''     
    
    fig, ax = plt.subplots(figsize=(16,8))

    fig.set_facecolor('white')
    ax.set_facecolor('white')

    ax.set_title(titulo, loc = 'left', fontsize=16, color='gray')

    for i, modelo in enumerate(modelos):
        plot_roc_curve(modelo , x, y, ax = ax) 
    
    ax.plot([0, 1], [0, 1], color = "red", ls ='--')

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_color('gray')   

    plt.xticks(fontsize= 14, color='gray')
    plt.yticks(fontsize= 14, color='gray')

    ax.set_xlabel(xlabel= 'Taxa de Falsos positivos'   , fontsize=14, color='gray')
    ax.set_ylabel(ylabel= 'Taxa de Verdadeiro positivos', fontsize=14, color='gray')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.show()
#********************************************************************* 