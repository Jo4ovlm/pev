import pandas as pd
import numpy as np

class myModel:
    
    def f(self, params, dist):
        return params[0]*dist**3 + params[1]*dist**2 + params[2]*dist + params[3]
    
    def retornaMedia(self, params, row, target):
        soma = 0                                  # inicializa a variavel que vai ser somada os valores de f()*media[j]
        sumDivisor = 0                            # inicaliza a variavel que vai ser somada os valores de f()
        for j in range(2,len(row)):               # Pra cada coluna de distancia na linha:
            if row.iloc[j] != 0:                  # Ignora as colunas da diagonal com a distancia pra propria pev
                peso = self.f(params, row.iloc[j])# faz o calculo com base nos parametros e na distancia da pev j
                soma += peso * target.iloc[j-2]   # faz soma += retorno da f(distancia entre pev[i] e pev[j]) * media de rendimento da pev j 
                sumDivisor += peso                # soma o retorno da f() a soma do divisor
        if sumDivisor == 0:                       # evitar divisão por zero
            return 0
        else:
            return soma / sumDivisor              # media é a soma dos resultados de f() ponderados pelas medias j, dividido pela soma sem ponderar
        
    
    def calculaErro(self, params, dataset_Treino, target):
        erro = np.zeros(len(dataset_Treino))          # inicializa o array que vai guardar o erro
        medias = np.zeros(len(dataset_Treino))        # inicializa o array que vai guardar as medias
        for i,row in dataset_Treino.iterrows():       # Pra cada linha do dataset:
            erro[i] = (self.retornaMedia(params, row, target) - target.iloc[i])**2  # erro = y_predito - y_real ao quadrado
        #print(f"params: {params}, peso_medio: {media}, real: {target.iloc[i]}, erro: {(media - target.iloc[i])**2}")
        return sum(erro)                              #retorna a soma do erro

            