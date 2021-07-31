import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

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
                     ax=ax)
    
    colorbar = ax.collections[0].colorbar

    num = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5 , 0.6, 0.7, 0.8, 0.9, 1.0]
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks(num)
    colorbar.set_ticklabels(num)
    plt.show()    
    
    