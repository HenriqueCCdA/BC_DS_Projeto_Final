# BC_DS_Projeto_Final

Repositório do projeto final do BootCamp Data Science da Alura 

## 1) Exploração e limepeza dos dados.

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


## 2) Seleção das variaveis explicativas.

A seleção das variaveis explicativas foi feita no notebook [selecao_variaveis.ipynb](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/tree/main/Notebooks/Exploratorios)

O escolha das variaveis explicativas foi feita utilizando a matriz de correlação para um valor de corte de **0.9**

O Dados com as colunas selecionados estão no arquivo [dados_sem_coor_acima_do_valor_de_corte.csv](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Dados/Tratados/dados_sem_coor_acima_do_valor_de_corte.csv). Para um consulta rápida das variaveis neste aquivo pode-se olhar o aquivo [dados_sem_coor_acima_do_valor_de_corte_colunas.csv](https://github.com/HenriqueCCdA/BC_DS_Projeto_Final/blob/main/Dados/Tratados/dados_sem_coor_acima_do_valor_de_corte_colunas.csv) com apenas os nomes da colunas.
