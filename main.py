import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PMC2 import *
from PMC1 import *
from RBF import *
from LVQ import *

# Obtem as entradas e saidas do dataset, a partir deste arquivo .data
dados_treinamento = pd.read_csv("mammographic_masses.data", sep=",").to_numpy()

# Remove as amostras onde alguma entrada esta com valor '?', ou seja, a entrada não está completa
dados_treinamento_completo = dados_treinamento[np.all(dados_treinamento != '?', axis=1)].astype(int)

mTreinamento = dados_treinamento_completo.shape[0]
nTreinamento = dados_treinamento_completo.shape[1]

np.random.shuffle(dados_treinamento_completo)

# Normaliza as entradas, em relação ao valor máximo de cada entrada entre as amostras
maxDados = np.amax(dados_treinamento_completo, axis=0)
dados_treinamento_completo = dados_treinamento_completo / maxDados

# Divide o dataset em 80% treinamento e 20% teste. Separa tambem as entradas das saidas em variáveis diferentes
div = int(0.8*mTreinamento) 
xTreinamento = dados_treinamento_completo[0:div, 0:nTreinamento-1]
yTreinamento = np.expand_dims(dados_treinamento_completo[0:div, nTreinamento-1], axis = 1)
# Transforma as saídas 0 em -1, pois a tangente hiperbólica que é a função de ativação da saída da rede produz valores entre -1 e 1 
yTreinamento[yTreinamento==0] = -1.0
xValidacao = dados_treinamento_completo[div:mTreinamento, 0:nTreinamento-1]
yValidacao = np.expand_dims(dados_treinamento_completo[div:mTreinamento, nTreinamento-1], axis = 1)

yValidacao[yValidacao==0] = -1.0

mVal = xValidacao.shape[0]

# Define os parâmetros de treinamento
n = 10**-3
epsilon = 10**-5

# CI = camadas intermediárias
print("Rede PMC com 2 CI - 50 e 25 neurônios")

w2, b2, w1, b1, w0, b0, eqmLista1 = treinamentoPMC2(xTreinamento, yTreinamento, [50,25,1], n, epsilon)

especificidades1, sensibilidades1 = validacaoPMC2(xValidacao, yValidacao, w2, b2, w1, b1, w0, b0) 

print("Rede PMC com 2 CI -  25 e 15 neurônios")

w2, b2, w1, b1, w0, b0, eqmLista2 = treinamentoPMC2(xTreinamento, yTreinamento, [25,15,1], n, epsilon)

especificidades2, sensibilidades2 = validacaoPMC2(xValidacao, yValidacao, w2, b2, w1, b1, w0, b0) 

print("Rede PMC com 2 CI - 15 e 10 neuronios")

w2, b2, w1, b1, w0, b0, eqmLista3 = treinamentoPMC2(xTreinamento, yTreinamento, [15,10,1], n, epsilon)

especificidades3, sensibilidades3 = validacaoPMC2(xValidacao, yValidacao, w2, b2, w1, b1, w0, b0) 

# Plota o gráfico de EQM por épocas
plt.figure()
plt.title("EQM por Épocas Treinamento - Rede PMC com 2 CI")
plt.ylabel("EQM")
plt.xlabel("Épocas")
plt.plot(eqmLista1, label='50 e 25 neurônios')
plt.plot(eqmLista2, label='25 e 15 neurônios')
plt.plot(eqmLista3, label='15 e 10 neurônios')
plt.legend(loc='best')
#plt.show()
plt.savefig("eqm_epoca_pmc2ci.png")

# Plota o gráfico da curva ROC
plt.figure()
plt.title("Curva ROC - Rede PMC 2 CI")
plt.ylabel("Sensibilidade (Taxa de verdadeiros positivos)")
plt.xlabel("1 - Especificidade (Taxa de falsos positivos)")
plt.plot(especificidades1, sensibilidades1, label='50 e 25 neurônios')
plt.plot(especificidades2, sensibilidades2, label='25 e 15 neurônios')
plt.plot(especificidades3, sensibilidades3, label='15 e 10 neurônios')
plt.legend(loc='best')
#plt.show()
plt.savefig("roc_pmc2ci.png")


print("Rede PMC com 1 CI - 50 neurônios")

w1, b1, w0, b0, eqmLista1 = treinamentoPMC1(xTreinamento, yTreinamento, [50,1], n, epsilon)

especificidades1, sensibilidades1 = validacaoPMC1(xValidacao, yValidacao, w1, b1, w0, b0) 


print("Rede PMC com 1 CI - 25 neurônios")

w1, b1, w0, b0, eqmLista2 = treinamentoPMC1(xTreinamento, yTreinamento, [25,1], n, epsilon)

especificidades2, sensibilidades2 = validacaoPMC1(xValidacao, yValidacao, w1, b1, w0, b0) 

print("Rede PMC com 1 CI - 15 neurônios")

w1, b1, w0, b0, eqmLista3 = treinamentoPMC1(xTreinamento, yTreinamento, [15,1], n, epsilon)

especificidades3, sensibilidades3 = validacaoPMC1(xValidacao, yValidacao, w1, b1, w0, b0) 

plt.figure()
plt.title("EQM por Épocas Treinamento - Rede PMC com 1 CI")
plt.ylabel("EQM")
plt.xlabel("Épocas")
plt.plot(eqmLista1, label='50 neurônios')
plt.plot(eqmLista2, label='25 neurônios')
plt.plot(eqmLista3, label='15 neurônios')
#plt.show()
plt.savefig("eqm_epoca_pmc1ci.png")

# Plota o gráfico da curva ROC
plt.figure()
plt.title("Curva ROC - Rede PMC 1 CI")
plt.ylabel("Sensibilidade (Taxa de verdadeiros positivos)")
plt.xlabel("1 - Especificidade (Taxa de falsos positivos)")
plt.plot(especificidades1, sensibilidades1, label='50 neurônios')
plt.plot(especificidades2, sensibilidades2, label='25 neurônios')
plt.plot(especificidades3, sensibilidades3, label='15 neurônios')
plt.legend(loc='best')
#plt.show()
plt.savefig("roc_pmc1ci.png")


print("Rede RBF - 50 neurônios")

w1, b1, w0, var0, eqmLista1 = treinamentoRBF(xTreinamento, yTreinamento, [50,1], n, epsilon)

especificidades1, sensibilidades1 = validacaoRBF(xValidacao, yValidacao, w1, b1, w0, var0) 

print("Rede RBF - 25 neurônios")

w1, b1, w0, var0, eqmLista2 = treinamentoRBF(xTreinamento, yTreinamento, [25,1], n, epsilon)

especificidades2, sensibilidades2 = validacaoRBF(xValidacao, yValidacao, w1, b1, w0, var0) 

print("Rede RBF - 15 neurônios")

w1, b1, w0, var0, eqmLista3 = treinamentoRBF(xTreinamento, yTreinamento, [15,1], n, epsilon)

especificidades3, sensibilidades3 = validacaoRBF(xValidacao, yValidacao, w1, b1, w0, var0) 

plt.figure()
plt.title("EQM por Épocas Treinamento - Rede RBF")
plt.ylabel("EQM")
plt.xlabel("Épocas")
plt.plot(eqmLista1, label='50 neurônios')
plt.plot(eqmLista2, label='25 neurônios')
plt.plot(eqmLista3, label='15 neurônios')
#plt.show()
plt.savefig("eqm_epoca_rbf.png")

# Plota o gráfico da curva ROC
plt.figure()
plt.title("Curva ROC - Rede RBF")
plt.ylabel("Sensibilidade (Taxa de verdadeiros positivos)")
plt.xlabel("1 - Especificidade (Taxa de falsos positivos)")
plt.plot(especificidades1, sensibilidades1, label='50 neurônios')
plt.plot(especificidades2, sensibilidades2, label='25 neurônios')
plt.plot(especificidades3, sensibilidades3, label='15 neurônios')
plt.legend(loc='best')
#plt.show()
plt.savefig("roc_rbf.png")

print("Rede LVQ")

w0, w0Classes = treinamentoLVQ(xTreinamento, yTreinamento, n)

validacaoLVQ(xValidacao, yValidacao, w0, w0Classes) 
