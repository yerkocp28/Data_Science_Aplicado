"""
Control N°1

Nombre: Yerko Carreño Pérez
RUT: 18.245.437-0


Pregunta N°2

"""

### Solucion Problema N°2 #####

import math 

acum=0.0001

print('Ingrese el valor de la altura h')
h= int(input())
for i in range(1,h):
     acum=0.0001*0.0001*acum+(h**3-3*math.pi*h)
print(f'La diferencia de volumenes entre un Cubo de lado {h} y un Cono de altura {h} con radio 3 es {acum}')


