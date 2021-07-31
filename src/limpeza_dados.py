import pandas as pd
import numpy as np

def pacientes_por_janela_ICU(dados):
  '''
  ------------------------------------------------------------------------------
  Função que conta se o paciente esta ou não na UTI para cada janela
  ------------------------------------------------------------------------------
  @param dados - dateFrame
  ------------------------------------------------------------------------------
  @return ICU_window_0_2  - numero de pacientes que foram para UTI janela 0-2
  @return ICU_window_2_4  - numero de pacientes que foram para UTI janela 2-4
  @return ICU_window_4_6  - numero de pacientes que foram para UTI janela 4-6
  @return ICU_window_6_12 - numero de pacientes que foram para UTI janela 6-12
  ------------------------------------------------------------------------------
  '''

  ICU_window_0_2 = dados.query("WINDOW == '0-2' and ICU==1")['PATIENT_VISIT_IDENTIFIER']
  ICU_window_2_4 = dados.query("WINDOW == '2-4' and ICU==1")['PATIENT_VISIT_IDENTIFIER']
  ICU_window_4_6 = dados.query("WINDOW == '4-6' and ICU==1")['PATIENT_VISIT_IDENTIFIER']
  ICU_window_6_12 = dados.query("WINDOW == '6-12' and ICU==1")['PATIENT_VISIT_IDENTIFIER']

  print('Numero de ICU igual a 1')
  print('Janela 0-2 :' ,len(ICU_window_0_2))
  print('Janela 2-4 :' ,len(ICU_window_2_4))
  print('Janela 4-6 :' ,len(ICU_window_4_6))
  print('Janela 6-12:',len(ICU_window_6_12))

  return ICU_window_0_2, ICU_window_2_4, ICU_window_4_6, ICU_window_6_12
  
def colunas_com_apenas_n_valores_unicos(dados, n=2, ci=13, cf=228):
  '''
  ------------------------------------------------------------------------------
  Mostra todas as colunas que so tem n dados unicos
  ------------------------------------------------------------------------------
  @param dados - dados
  @param n     - numero de dados unicos desejados
  @param ci    - coluna inicial
  @param cf    - coluna final
  ------------------------------------------------------------------------------
  '''

  print(f"Colunas com apenas {n} valores unicos:")
  for name in dados.columns[ci:cf]:
    if len( dados[name].unique()) < n + 1:
      print(f'{name:20} ->', dados[name].unique())

def retira_paciente_primeira_janela(matriz):
  '''
  ------------------------------------------------------------------------------
  Retira da base de dados os pacientes que foram para UTI na pimeira janela
  0-2
  ------------------------------------------------------------------------------
  @param matriz - dados brutos
  ------------------------------------------------------------------------------
  @return matriz sem o pacientes que que foram para UTI na pimeira janela
  ------------------------------------------------------------------------------
  '''

  window = matriz.query("WINDOW=='0-2' and ICU==1")['PATIENT_VISIT_IDENTIFIER']

  matriz = matriz.query("PATIENT_VISIT_IDENTIFIER not in @window")

  return matriz

def submatriz_preenchimento(submatriz):
  '''
  ------------------------------------------------------------------------------
  Preenche as variaveis continuas utilizando. Esta função trabalho com submatriz
  da matriz principal
  ------------------------------------------------------------------------------
  @param submatriz - submatriz agrupapor paciente e UCIs
  ------------------------------------------------------------------------------
  @return submatriz preenchida quando 
  ------------------------------------------------------------------------------
  '''
  
  submatriz_var_continuas = submatriz.iloc[:, 13:-2]
  submatriz_var_categorica_inicio = submatriz.iloc[:, :13]
  submatriz_var_categorica_final  = submatriz.iloc[:,-2:]

  # preenchendo os valores NaN 

  submatriz_var_continuas = submatriz_var_continuas.fillna(method='ffill')\
                                                   .fillna(method='bfill')

  submatriz_preenchida = pd.concat([submatriz_var_categorica_inicio, 
                                      submatriz_var_continuas,  
                                      submatriz_var_categorica_final],
                                      ignore_index=False, axis=1)
  
  return submatriz_preenchida


def preenchendo_var_continuas(matriz):

  '''
  ------------------------------------------------------------------------------
  Preenche as variaveis continuas utilizando os somente os dados para quando 
  ITU = 0. 
  ------------------------------------------------------------------------------
  @param matriz - dados brutos
  ------------------------------------------------------------------------------
  @return matriz sem o pacientes que que foram para UTI na pimeira janela
  ------------------------------------------------------------------------------
  '''

  matriz_preenchida = matriz.groupby(['PATIENT_VISIT_IDENTIFIER', 'ICU'],
                                     as_index=False).apply(submatriz_preenchimento)
  matriz_preenchida = matriz_preenchida.reset_index().drop(['level_0', 
                                                            'level_1'], axis=1)

  return matriz_preenchida


def uma_linha_por_paciente(submatriz):
  '''
  ------------------------------------------------------------------------------
  Reduz todas as janelas do paciente a apenas um linha. Os valores são pegos
  da janela 0-2
  ------------------------------------------------------------------------------
  @param submatriz - Submatriz agrupa por pacientes
  ------------------------------------------------------------------------------
  @return Retorna os valores da janela 0-2 com a informa se o paciente foi ou 
  não para a UTI
  ------------------------------------------------------------------------------
  '''

  if np.any(submatriz['ICU']):
    submatriz.iloc[:,-1] = 1

  return submatriz.iloc[0,:]