# Repositório do projeto final do BootCamp Data Science da Alura

## 1) Introdução

Logo no início da pandêmia de **Covid-19** em 2020 um fator que ficou claro foi a alta demanda de UTIs para o tratamento dos doentes. Sem um tratamento adquado a **taxa de mortalidade** aumenta de maneira alarmante.

UTIs são unidades médicas extramamente custosas. Por isso é inviavel que se aumente o número de UTIs de maneira indiscrimina. Tendo isso em vista, uma ferramente que consiga prever a necessidade de internação em UTI de um paciente através de dados clinicos seria uma grande ferramenta para atentar ajustar a disponibilidades de UTIs horas antes delas serem realmente necessarias para salvar a vida do paciente.  

### 1.2) Objetivo

Neste estudo tem como objetivo escolher um modelo de Machine Learning (ML) que consiga prever se um paciente irá precisar ou não de UTI através do dados clinicos de horas antes da entrada do paciente na UTI.

### 1.3) Metodologia

Através de disponibilidados pelo hospital [Sírio-Libanês](https://www.kaggle.com/S%C3%ADrio-Libanes/covid19) serão rodas 6 modelos de Machine Learning (ML). Para escolha do melhor modelos foi utlizado a tecinica de validação crusada (Cross Validation) e ajuste de hyperparametros. 


## 2) Dados brutos

Os dados brutos foram tirados do projeto do kaggle [Sírio-Libanês](https://www.kaggle.com/S%C3%ADrio-Libanes/covid19). O dados podem ser ser achado neste repositorio no arquivo [Kaggle_Sirio_Libanes_ICU_Prediction.xlsx](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Dados/Brutos/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx)

## 3) Exploração e limepeza dos dados.

A exploração e limpeza dos dados foram feitas no notebook foi feita no notebook [explaracao_limpezada.ipynb](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Notebooks/Exploratorios/explaracao_limpezada.ipynb)

O tratamento foi basicamente:

1. Retiradas de pacientes que já foram para **UTI** na primeira janela.
2. Preenchimento de dados **NaN** nas colunas de variaveis continuas.
3. **Descarte** de qualquer linha que ainda tenha valores **NaN** após a **etapa 2**.
4. Apenas informoções dos pacientes apenas da primeira janela. Assim temos uma **linha por paciente**
5. **Descarte** de algumas colunas que são funções de outras (**DIFF** e **DIF_REL**)
6. Transformação da colunas **AGE_PERCENTIL** para numérica
7. **Descarte** das colunas **WINDOW** e **PATIENT_VISIT_IDENTIFIER**
8. Arquivo final tratado salvo.

O Dados limpos estão no arquivo [dados_tratados_por_paciente.csv](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Dados/Tratados/dados_tratados_por_paciente.csv). Para um consulta rápida das variaveis neste aquivo pode-se olhar o aquivo [dados_tratados_por_paciente_colunas.csv](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Dados/Tratados/dados_tratados_por_paciente_colunas.csv) com apenas os nomes da colunas.

## 4) Seleção das variaveis explicativas.

A seleção das variaveis explicativas foi feita no notebook [selecao_variaveis.ipynb](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/tree/main/Notebooks/Exploratorios)

O escolha das variaveis explicativas foi feita utilizando a matriz de correlação para um valor de corte de **0.9**

O Dados com as colunas selecionados estão no arquivo [dados_sem_coor_acima_do_valor_de_corte.csv](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Dados/Tratados/dados_sem_coor_acima_do_valor_de_corte.csv). Para um consulta rápida das variaveis neste aquivo pode-se olhar o aquivo [dados_sem_coor_acima_do_valor_de_corte_colunas.csv](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Dados/Tratados/dados_sem_coor_acima_do_valor_de_corte_colunas.csv) com apenas os nomes da colunas.

## 5) Modelos de machine learning (ML)

Foram escolhidos 6 modelos de ML, são eles: 

* DummyClassifier (baseline)
* LogisticRegression
* DecisionTreeClassifier
* Forest Tree
* Support Vector Machine
* KNeighbors

Para a escolha do melhor modelo foi feita uma busca de hyperparametros. Estas buscas foram feitas pelo GridSearchCV ou pelo RandomizedSearchCV. Para a Cross Validation foi usado **RepeatedStratifiedKFold** com **5** divições de **10** repetições. O parametro utilizado para avaliação do modelo foi **ROC_AUC** e quantidade de **falsos negativos** (FN).

