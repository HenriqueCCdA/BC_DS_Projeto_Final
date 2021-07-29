import matplotlib.pyplot as plt
import seaborn as sns

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