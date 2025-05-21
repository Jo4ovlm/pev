import pandas as pd
import numpy as np
from math import log

class myModel:
    X_treino = pd.DataFrame()
    
    def f(self, params, dist):
        return params[0]*dist**4 + params[1]*dist**3 + params[2]*dist**2 + params[3]*dist + params[4]
    
    def getPevIndex(self, nome_coluna):
        return float(''.join(filter(str.isdigit, nome_coluna))) -1
    
    def retornaMediaTreino(self, params, row, target):
        soma = 0
        sumDivisor = 0
        for j in range(2, len(row)):
            if row.iloc[j] != 0:
                peso = self.f(params, row.iloc[j])
                nome_coluna = row.index[j]         # Ex: 'pev5'
                idx_target = self.getPevIndex(nome_coluna)  # → 5.0
                if idx_target in target:
                    soma += peso * target[idx_target]
                    sumDivisor += peso
        return 0 if sumDivisor == 0 else soma / sumDivisor             # media é a soma dos resultados de f() ponderados pelas medias j, dividido pela soma sem ponderar
        
    
    def calculaErro(self, params, dataset_Treino, target):
        self.X_treino = dataset_Treino
        erro = np.zeros(len(dataset_Treino))          # inicializa o array que vai guardar o erro
        medias = np.zeros(len(dataset_Treino))        # inicializa o array que vai guardar as medias
        j =0                                          # j = indice sequencial da posição no dataset
        for i,row in dataset_Treino.iterrows():       # Pra cada linha do dataset:
            erro[j] = (self.retornaMediaTreino(params, row, target) - target.iloc[j])**2 # erro = y_predito - y_real ao quadrado
            j += 1      
        #print(f"params: {params}, peso_medio: {media}, real: {target.iloc[i]}, erro: {(media - target.iloc[i])**2}")
        return sum(erro)                              #retorna a soma do erro
    
    
    def calculaMediasPredicao(self, params, row, y_treino):
        lat = row["latitude"]
        long = row["longitude"]
        soma = 0
        divisor = 0
        row = row.drop("latitude")
        row = row.drop("longitude")
        for j in range(0, len(row)):
            lat #latitude do j
            long #longitude do j
            dist = row.iloc[j]
            if dist != 0:
                peso = self.f(params, dist)
                idx_pev = self.getPevIndex(row.index[j])
                if idx_pev in y_treino:
                    m_j = y_treino[idx_pev]              # Usa o índice, não posição
                    soma += peso * m_j
                    divisor += peso

        return 0 if divisor == 0 else soma / divisor
        
    
    def preve(self, params,dataset_teste, y_treino):
        y_predito = np.zeros(len(dataset_teste))
        for i,row in dataset_teste.iterrows():
            y_predito[i] = self.calculaMediasPredicao(params,row, y_treino)
            
        return y_predito

            