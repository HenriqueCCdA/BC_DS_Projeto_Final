![](https://img.shields.io/github/last-commit/HenriqueCCdA/bootCampAluraDataScience?style=plasti&ccolor=blue)
![](https://img.shields.io/badge/Autor-Henrique%20C%20C%20de%20Andrade-blue)

![](https://play-lh.googleusercontent.com/E5OY3A9Nf-XieZN5Ah6KfPIDbFpLR_j5fFOLbl-aYDrRiFAvensqRJjZpWFRA_yyNg)


# Repositório do projeto final do BootCamp Data Science da Alura

---
---
# Análise


---
## 1) Introdução

Logo no início da pandêmia de **Covid-19** em 2020 um fator que ficou claro foi a alta demanda de UTIs para o tratamento dos doentes. Sem um tratamento adquado a **taxa de mortalidade** aumenta de maneira alarmante.

UTIs são unidades médicas extramamente custosas. Por isso é inviavel que se aumente o número de UTIs de maneira indiscrimina. Tendo isso em vista, uma ferramente que consiga prever a necessidade de internação em UTI de um paciente através de dados clinicos seria uma grande ferramenta para atentar ajustar a disponibilidades de UTIs horas antes delas serem realmente necessarias para salvar a vida do paciente.  

### 1.2) Objetivo

Este estudo tem como objetivo escolher um modelo de Machine Learning (ML) que consiga prever se um paciente irá precisar ou não de **UTI** através de dados clínicos apenas.

### 1.3) Metodologia

Através dos dados disponibilidados pelo hospital [Sírio-Libanês](https://www.kaggle.com/S%C3%ADrio-Libanes/covid19) serão treinados **6** modelos de Machine Learning (ML). Para escolha do melhor modelo foi utlizado a têcnica de validação crusada (**Cross Validation**), ajuste de hyperparametros (**hyperparameter tuning**) e seleção de variaveis explicativas (**feature selection**). Os modelos foram testados em **4** conjutos de variaveis explicativas possiveis. 

---
## 2) Dados brutos

Os dados brutos foram tirados do projeto do kaggle [Sírio-Libanês](https://www.kaggle.com/S%C3%ADrio-Libanes/covid19). Este dados podem ser ser achados neste repositório no [aqui](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Dados/Brutos/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx).

---
## 3) Exploração e limepeza dos dados.

A exploração e limpeza dos dados foi feita [neste notebook](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Notebooks/Exploratorios/explaracao_limpezada.ipynb)

O tratamento foi basicamente:

1. Retirada de pacientes que já foram para UTI na primeira janela de atendimento (0-2h).
2. Preenchimento de dados **NaN** nas colunas das variaveis continuas.
3. Descarte de qualquer linha que ainda tenha valores **NaN** após a **etapa 2**.
4. Manter apenas informoções dos pacientes da primeira janela. Assim temos uma linha por paciente.
5. Transformação da colunas AGE_PERCENTIL para numérica.
6. Descarte das colunas WINDOW e PATIENT_VISIT_IDENTIFIER.
7. Descarte das colunas com variância menor que 0.01.

Os Dados limpos estão no arquivo [dados_tratados_por_paciente.csv](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Dados/Tratados/dados_tratados_por_paciente.csv). Para uma consulta rápida das variaveis neste aquivo pode-se olhar o aquivo [dados_tratados_por_paciente_colunas.csv](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Dados/Tratados/dados_tratados_por_paciente_colunas.csv) com apenas os nomes da colunas (possível variaveis + o alvo(ICU)).

---
## 4) Seleção das variaveis explicativas.

Temos 4 seleções de variaveis explicativas, são elas:

* Seleção através da matriz de correlação. O módulo do limiar de correção foi **0.9**. Após o procedimentos chegou-se a **41** variaveis explicativas. Chamaremos esse conjuto de dados de **df_coor_41**.
  * [Notebook](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Notebooks/Selecao_variaveis/selecao_variaveis.ipynb). 
  * [Base](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Dados/Tratados/dados_sem_corr_acima_do_valor_de_corte.csv) e [variaveis selecionas](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Dados/Tratados/dados_sem_corr_acima_do_valor_de_corte_colunas.csv). 

* Seleção através da matriz de correlação aliada com a técnica **recursive feature elimination (RFE)**.  O limiar de correção foi novamente **0.9** e o modelo usado no **RFE** foi o **LogisticRegression**. Um conjuto de **20** e **30** variáveis foram escolhidos. Chamaremos esses conjuto de dados de **df_RFE_20** e **df_RFE_30**
  * [Notebook](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Notebooks/Selecao_variaveis/selecao_variaveis_sklearn.ipynb). 
  * [Base 20](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Dados/Tratados/dados_rfe20.csv), [base 30](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Dados/Tratados/dados_rfe30.csv), [variaveis selecionas da base 20](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Tratados/dados_rfe20_colunas.csv) e
  [variaveis selecionas da base 30](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Dados/Tratados/dados_rfe30_colunas.csv). 

* Seleção atravez da lib [featurewiz](https://github.com/AutoViML/featurewiz). Após o procedimentos chegou-se a **25** variaveis explicativas. Chamaremos esse conjuto de dados de **df_featurewiz_25**
  * [Notebook](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Dados/Tratados/dados_featurewiz.csv). 
  * [Base](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Dados/Tratados/dados_sem_corr_acima_do_valor_de_corte.csv) e [variaveis selecionas](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Dados/Tratados/dados_featurewiz_colunas.csv).

---
## 5) Modelos de machine learning (ML)

Foram escolhidos 6 modelos de ML, são eles: 

* DummyClassifier (baseline)
* LogisticRegression
* DecisionTreeClassifier
* Random Forest Tree
* Support Vector Machine
* KNeighbors

Para a escolha do melhor modelo foi feita uma busca de hyperparametros. A buscas foi evetuda pelo GridSearchCV ou pelo RandomizedSearchCV. Para a **Cross Validation** foi usado **RepeatedStratifiedKFold** com **5** divisões de **10** repetições. O parâmetro utilizado para avaliação do modelo foi a **ROC_AUC**. A quantidade de **falsos negativos** (FN) também é uma métrica crítica da nosso problema. Um bom modelo tem que ter um **ROC_AUC** alto e um **FN** baixo. O **FN** neste caso significa mandar um paciente que precisaria de UTI para casa, o que pode acarretar a morte do paciente.

Os todos os **conjuto de dados (dataset)** foram dividos em um dataset para **validação cruzada (Cross Validation)** e um de **validação**. A divisão ficou **351** para **validação cruzada** e **36** para a **validação**. A escolha do método se deu tanto pela avaliação do desempenho do modelo no conjuto de dados de **validação** quanto nas métricas médias e desvio padrão dos testes da **validação cruzada**.

Iniciamentel foi utilização **df_coor_41** para uma analise mais manual do treinamento, o que pode se encontrado [Neste notebbok](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Notebooks/ML/treinamentos_dados1.ipynb). Nele chega-se a conlusão que o melhor modelo é **Random Forest Tree** com **ROC_AUC** de **0.78** no dataset de **validação** e média de **0.81** na **validação cruzada**.

Após isto, em um outro [notebook](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Notebooks/ML/treina_diferentes_var_explicativas.ipynb), foi analisado as outras **3** bases restantes. Nele novamente concluimos que **Random Forest Tree** é a melhor opção. Além disso a melhor seleção de variaveis foi **df_featurewiz_25**, ela obteve **0.835913** no dataset de **validação** e **0.817647** na média dos teste na **validação cruzada**

---
# 6) Modelo final 

O modelo final excolhido com ja dito foi **Random Forest Tree** com o conjuto de variveis explicativas do **df_featurewiz_25** ([lista das variaveis](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Dados/Tratados/dados_featurewiz_colunas.csv)). O modelo foi treinado utilizando todas as **351** amostras [neste notebook](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Notebooks/ML/Treinamento_modelo_final.ipynb). O modelo foi salvo em disco com o auxilio da lib **joblib** no arquivo [modelo_final.sav](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Modelo_ML/modelo_final.sav). 


---
# 7) Conclusão

Finalmente agora temos um modelo de **ML** capaz de predizer com um certo grau de acuracia se o paciente irá precisar ao não de **UTI** apenas com os dados clínicos de horas antes. Assim o objetivo do estudo foi cumprido.

---
---

# Informações importantes do repositório:
---

---
* Estrutura de diretorios do repositório:

```
└──Dados
|  ├── Brutos - Dados brutos
|  └── Tratrados - Dados Tratados 
|
└──Notebooks
|  ├── Explaratorios - Notebooks exploatorios e de limpeza
|  ├── Seleção_variaveis - Notebbooks de seleção de variaveis 
|  └── ML - Notebbooks com os treinamentos dos modelos 
|           
└── src - Arquivos fontes python   
|           
└── Modelo_ML - Modelos de ML salvos  
```

---
* Ordem indica de leitura dos notebooks:

1. [Limpeza e Exploração dos dados](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Notebooks/Exploratorios/explaracao_limpezada.ipynb)
2. [Seleção de variaveis - Método 1](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Notebooks/Selecao_variaveis/selecao_variaveis.ipynb)
3. [Seleção de variaveis - Método 2](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Notebooks/Selecao_variaveis/selecao_variaveis_sklearn.ipynb)
4. [Seleção de variaveis - Método 3](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Notebooks/Selecao_variaveis/selecao_variaveis_featurewiz.ipynb)
5. [Primeiro Treinamento na base df_coor_41](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Notebooks/ML/treinamentos_dados1.ipynb)
6. [Treinamento com a 4 bases](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Notebooks/ML/treina_diferentes_var_explicativas.ipynb)
7. [Treinamento do modelo final](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Notebooks/ML/Treinamento_modelo_final.ipynb)

---
* Libs utilizadas:
  * matplotlib   -> 3.3.4
  * sklean       -> 0.24.2
  * pandas       -> 1.2.4
  * scipy        -> 1.6.0
  * numpy        -> 1.20.2
  * sns          -> 0.11.1
  * freaturewiz  -> 0.0.42
  * joblib       -> 1.0.1





