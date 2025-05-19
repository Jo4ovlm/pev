import numpy as np
from math import radians, sin, cos, sqrt, atan2

class criadorTabelaDistancia:
    coordenadas = []
    
    def __init__(self,coordenadas):
        self.coordenadas = coordenadas
        
        
    def d(self,lat1,long1,lat2,long2):
        R = 6371  # Raio da Terra em km
        lat1, lon1, lat2, lon2 = map(radians, [lat1, long1, lat2, long2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c # Retorna distancia em km
        
    def calculaDistancias(self,x,y):
        dist = np.zeros(len(self.coordenadas))
        
        for i in range(0,len(self.coordenadas)):
            lat2 = self.coordenadas.iloc[i]["latitude"]
            long2= self.coordenadas.iloc[i]["longitude"]
            dist[i] = self.d(x,y,lat2,long2)

        
        return dist