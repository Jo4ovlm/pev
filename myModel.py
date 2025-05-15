import pandas as pd
import numpy as np

class myModel:
    
    def f(self, params, dist):
        return params[0]*dist**3 + params[1]*dist**2 + params[2]*dist + params[3]
    
    def fit(self, params, dataset_Treino, target):
        erro = np.zeros(len(dataset_Treino))
        medias = np.zeros(len(dataset_Treino))
        for i,row in dataset_Treino.iterrows():
            soma = 0
            sumDivisor = 0
            for j in range(len(row)):
                valor = self.f(params, row[j])
                soma += valor * target[i]
                sumDivisor += valor
            if sumDivisor == 0:  # evitar divisão por zero
                media = 0
            else:
                media = soma / sumDivisor
            medias[i] = media
            erro[i] = (media - target[i])**2
        return sum(erro)

            