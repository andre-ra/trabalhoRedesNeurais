import numpy as np
import matplotlib.pyplot as plt
import time



def treinamentoPMC1(x, y , numNeuronios, n = 0.01, epsilon = 10**-7, th = 0):

    tempo_antes = time.time()
    

    # Treinamento

    m = x.shape[0]
    # Inicializa os pesos com valores aleatórios pequenos
    w0 = 0.1*np.random.rand(numNeuronios[0], x.shape[1])
    b0 = 0.1*np.random.rand(numNeuronios[0], 1)
    w1 = 0.1*np.random.rand(numNeuronios[1], w0.shape[0])
    b1 = 0.1*np.random.rand(numNeuronios[1], 1)
    # Inicializa as outras variáveis
    yEst = np.zeros((m, y.shape[1]))
    eqm = 1
    eqmAnt = 0
    epocas = 0
    clip = 0.5
    eqmLista = []

    # Regra de parada
    while (abs((eqm - eqmAnt)) > epsilon):

        # Embaralha o dataset de treino a cada época, garantindo a mesma ordem de x e y (ou seja, entradas junto com a saída)
        randomize = np.arange(len(x))
        np.random.shuffle(randomize)
        x = x[randomize] 
        y = y[randomize]
        
        # Zera as derivadas e os vieses (biases)
        dW1 = 0
        dB1 = 0
        dW0 = 0
        dB0 = 0    
            
        for i in range(0,m):
            
            # Feedforward da amostra

            # Adicinar o bias desta forma é o mesmo que adicionar -1 na matriz dos valores de saída da camada anterior (-b0 = +(-1)b0)
            u0 = np.dot(w0, np.expand_dims(x[i], axis=1)) - b0
            a0 = np.tanh(u0)

            u1 = np.dot(w1, a0) - b1
            a1 = np.tanh(u1)

            # Trocar o formato de (y.shape[1], 1) para um vetor de tamanho y.shape[1]
            yEst[i] = np.squeeze(a1, axis=0)

            # Calcula o erro na amostra (soma dos erros nos neuronios de saida, neste caso não
            # faz diferença pois só há um neurônio e, portanto, só uma saída de erro)
            e = np.sum(y[i] - yEst[i])

            # Calcula as derivadas dos pesos e dos biases, baseado no erro nesta amostra
            # Derivada de tanh(x) = 1 -x²
            delta1 = e*(1-np.power(u1, 2))                
            # Clip na variaçao do gradiente para impedir os pesos de irem para infinito
            dW1 += np.clip(n*delta1*(a0.T), -clip, clip)
            dB1 += np.clip(n*delta1*-1, -clip, clip)

            delta0 = np.sum(delta1*w1)*(1-np.power(u0, 2))
            dW0 += np.clip(n*np.dot(delta0, (x[i].reshape(-1, 1).T)), -clip, clip)
            dB0 += np.clip(n*delta0*-1, -clip, clip)
                
        # Atualiza as derivadas
        w1 += dW1/m
        b1 += dB1/m

        w0 += dW0/m
        b0 += dB0/m

        # Salva o erro quadrático médio anterior
        eqmAnt = eqm

        # Calcula o erro na amostra (soma dos erros nos neuronios de saida, neste caso não
        # faz diferença pois só há um neurônio e, portanto, só uma saída de erro)
        e = np.sum(y - yEst, axis=1)

        # Calcula o erro quadrático médio
        # Eqm = soma(e²)/(2*m)
        eqm = np.sum(np.power(e, 2))/(2*m)
        eqmLista.append(eqm)

        epocas +=1


    print("Épocas: " + str(epocas))
    print("EQM final: " + str(eqm))

    # Obtem a duração do treinamento
    tempo_depois = time.time() 
    print("Tempo de processamento: " + str(tempo_depois - tempo_antes))


    return w1, b1, w0, b0, eqmLista

def validacaoPMC1(xVal, yVal, w1, b1, w0, b0):

    # Inicializa as variáveis
    m = xVal.shape[0]
    yEst = np.zeros((m, w1.shape[0])) 

    for i in range(0,m):

        # Feedforward nas amostras de validação
        u0 = np.dot(w0, np.expand_dims(xVal[i], axis=1)) - b0
        a0 = np.tanh(u0)

        u1 = np.dot(w1, a0) - b1
        a1 = np.tanh(u1)

        yEst[i] = np.squeeze(a1, axis=0)

    # Inicializa as variáveis para analise de performance da validação (acurária e curva roc)
    quantTh = 1001
    sensibilidades = []
    especificidades = []
    minTh = -1
    maxTh = 1
    # Obtem a lista de thresholds a serem analisados
    ths = np.linspace(maxTh, minTh, num=quantTh)

    # Para cada threshold contido na lista de thresholds
    for th in ths:
        # Classifica as saídas das amostras
        yEstI = 1*(yEst>th)
        yEstI[yEstI==0] = -1
        dif = yVal - yEstI
        
        # Calcula a sensibilidade, a especificidade e a acurária e armazena em uma lista
        falsoNeg = np.sum(1*(dif == 2))
        falsoPos = np.sum(1*(dif == -2))
        verdadeiroPos = np.sum(1*((yVal == 1) & (dif == 0)))
        verdadeiroNeg = np.sum(1*((yVal == -1) & (dif == 0)))
        # Calcula a sensibilidade (ou taxa de verdadeiros positivos)
        sensibilidades.append(verdadeiroPos / (verdadeiroPos + falsoNeg))
        # Calcula 1 - especificidade (ou taxa de falsos positivos), usada na curva ROC
        especificidades.append(1 - verdadeiroNeg / (verdadeiroNeg + falsoPos))

    # Calcula a area embaixo da curva ROC (area under curve)
    # Esta área entre duas amostras pode ser calculada como a area de um trapézio onde as bases são
    # os valores de sensibilidade (ou seja, eixo y) de duas amostras consecutivas e a altura a 
    # diferença entre valores de 1 - especificidade (eixo y)
    auc = 0
    auc09 = 0
    for i in range(0, len(sensibilidades)-1):
        base =  sensibilidades[i+1] + sensibilidades[i]
        altura = especificidades[i+1] - especificidades[i]
        auc += base*altura/2
        # Obtem a AUC da região onde a sensibilidade é alta, acima de 90%
        if sensibilidades[i] > 0.9:
            auc09 += base*altura/2


    print("AUC: " + str(auc))
    print("AUC0.9: " + str(auc09))

    j = 0
    while (sensibilidades[j] < 0.95):
        j +=1

    print("Sensibilidade: " + str(sensibilidades[j]))
    print("Especificidade: " + str(1 - especificidades[j]))
    print("Threshold: " + str(ths[j]))

    return especificidades, sensibilidades
    