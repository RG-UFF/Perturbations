# Modos Quasi-Normais de Estrelas de Nêutrons e Buracos Negros

Este reposítório contém os notebooks desenvolvidos no estudo de oscilações de objetos compactos e extração das frequências dos modos quasi-normais emitidas na forma de ondas gravitacionais. Esse trabalho faz parte do projeto de mestrado da aluna Isabella Ramos de Souza Nunes (isabellarsn@id.uff.br - UFF) sob orientação da Profa. Dra. Raissa Fernandes Pessoa Mendes (rfpmendes@id.uff.br - UFF).

Link para da [Dissertação](por o link)

## Notebooks - Wolfram Mathematica

*  **Axial_perturbations.nb:** Destinado ao estudo das equações axiais que descrevem perturbações em buracos negros e estrelas de nêutrons;
*  **Polar_perturbations.nb:** Destinado ao estudo das equações polares que descrevem perturbações em buracos negros e estrelas de nêutrons;
*  **MQNSofBN.nb:** Cálculo dos coeficientes da relação de recorrência para análise dos MQNs de buracos negros no domínio das frequências;
*  **MQNSofEN.nb:** Análise das equações sobre as condição de contorno da origem e na superfície da estrela e cálculo dos coeficientes da relação de recorrência para análise dos MQNs de estrela de nêutrons no domínio das frequências.

  ## Notebooks - Python
  
  *  **BN_DomFreq.ipynb:** Extração dos MQNs de um buraco negro pelo método de frações continuadas no domínio das frequências;
  *  **BN_DomTemp.ipynb:** Extração dos MQNS de um buraco negro pelo método de DFT e pelo método de fit no domínio temporal;
  *  **EN_DomFreq.ipynb:** Extração dos MQNs de uma estrela de nêutrons pelo método de frações continuadas no domínio das frequências;
  *  **EN_DomTemp.ipynb:** Extração dos MQNS de uma estrela de nêutrons pelo método de DFT e pelo método de fit no domínio temporal;
  *  **Cont_Frac.ipynb:** Função responsável pela resolução da fração continuada para uma estrela de nêutrons;
  *  **Eq_timeD.py:** Definição das equações necessárias para executar a evolução temporal das equações de perturbação no domínio temporal;
  *  **Eq_freqD.py:** Definição do lado direito das equações de perturbação e funções necessárias para a itegração no domínio das frequências;
  *  **EOS.py:** Notebook onde são criadas as equações de estado e parâmetros correspondentes responsáveis por caracterizar uma estrela;
  *  **Background.py:** Definição das equações de fundo, como por exemplo as equações de TOV;
  *  **RK.py:** Definições do Runge-Kutta de terceira e quarta ordem;
  *  **Constants.py:** Constantes utilizadas ao longo dos cálculos;

## Demais arquivos
* **diffgeo.m:** Pacote do Wolfram Mathematica usado para cálculos da Relatividade Geral.
* **Tabela EOS:** Conjunto de dados que fornecem diferentes equações de estado.




  
