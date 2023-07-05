import numpy as np
import matplotlib.pyplot as plt
import time



def treinamentoLVQ(x, y , n = 0.01, epsilon = 10**-8):

    tempo_antes = time.time()
    

    # Treinamento
    m = x.shape[0]

    # Normalizando as amostras
    norma = np.expand_dims(np.linalg.norm(x, axis=1), axis=1)
    x = x / norma

    # Pega o indice de cada elemento unico da saida
    indices = np.unique(y, return_index=True)
    # Inicializa cada neurônio com o primeiro valor da primeira amostra de cada classe
    w0 = np.copy(x[indices[1]])
    w0Classes = indices[0]
    #Treinamento parte inicial

    # Inicializa as variáveis
    epocas = 0
    w0Ant = np.copy(w0) + 1

    while (np.sum(np.power(w0Ant - w0,2)) > epsilon):

        # Embaralha o dataset de treino a cada época, garantindo a mesma ordem de x e y (ou seja, entradas junto com a saída)
        randomize = np.arange(len(x))
        np.random.shuffle(randomize)
        x = x[randomize] 
        y = y[randomize]
        w0Ant = w0
        falhas = 0

        for i in range(0,m):

            # Obtem o neurônio com menor distancia Euclidiana da amostra (A menor distância ao quadrado é a menor distância
            # portanto, na prática não a necessidade de tirar a raiz aqui)
            d0 = np.sum(np.power(x[i] - w0,2), axis = 1).reshape(-1, 1)
            vencedor = np.argmin(d0)
            # Ajusta os pesos
            if y[i] == w0Classes[vencedor]:
                w0[vencedor] += n*(x[i] - w0[vencedor])
            else:
                w0[vencedor] -= n*(x[i] - w0[vencedor])
                falhas += 1

            # Normaliza os pesos ajustados
            norma = np.expand_dims(np.linalg.norm(w0, axis=1), axis=1)
            w0 = w0 / norma

        epocas += 1

    print("Épocas: " + str(epocas))
    print("F: " + str(falhas))

    # Obtem a duração do treinamento
    tempo_depois = time.time() 
    print("Tempo de processamento: " + str(tempo_depois - tempo_antes))


    return w0, w0Classes

def validacaoLVQ(xVal, yVal, w0, w0Classes):

    # Inicializa as variáveis
    m = xVal.shape[0]
    yEst = np.zeros((m, 1)) 

    for i in range(0,m):

        # Obtem o neurônio com menor distancia Euclidiana da amostra
        d0 = np.sum(np.power(xVal[i] - w0,2), axis = 1).reshape(-1, 1)
        vencedor = np.argmin(d0)
        yEst[i] = w0Classes[vencedor]


    dif = yVal - yEst
    
    # Calcula a sensibilidade e a especificidade
    falsoNeg = np.sum(1*(dif == 2))
    falsoPos = np.sum(1*(dif == -2))
    verdadeiroPos = np.sum(1*((yVal == 1) & (dif == 0)))
    verdadeiroNeg = np.sum(1*((yVal == -1) & (dif == 0)))
    sensibilidade = verdadeiroPos / (verdadeiroPos + falsoNeg)
    especificidade = verdadeiroNeg / (verdadeiroNeg + falsoPos)
                                     
    print("Sensibilidade: " + str(sensibilidade))
    print("Especificidade: " + str(1 - especificidade)) 
    