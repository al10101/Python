'''
Autor: Alberto Cabrera
Contacto: albertocabja@gmail.com
Realiza el calculo scf (restringido) y traza la funcion de onda al cuadrado 
para la molecula de H2 con la base STO-3G. Obtiene una caja de 3d con N
vectores distribuidos uniformemente y escoje ponderadamente con el valor de la
densidad de cada uno. Para reproducir el momento dipolar con la simulacion, 
la elección de numeros aleatorios asegurara que haya la misma cantidad
de puntos de un lado que de otro, tomando como "espejo" al eje Z.
'''

# Librerias para el calculo SCF
import numpy as np
from scf import main as SCF

# Librerias para trazar la grafica
import matplotlib.pyplot as plt

def main():

	# Numero de electrones en el sistema
    N = 2
    
    # Coordenadas cartesianas de los nucleos
    coordinates = np.array([[0.0, 0.0, -0.7],
                            [0.0, 0.0,  0.7]])
    lim = 3
    
    # Cargas nucleares
    Z = [1, 1]
    
    # Valor zeta para el atomo de H en la base STO-3G
    zeta = [1.24, 1.24]
    
    # Exponentes a y coeficientes d,xp para expresar la base como funcion 
    # de zeta
    d = np.array([[0.444635, 0.535328, 0.154329]])
    a = np.array([[0.109818],
                  [0.405771], 
                  [2.22766]])
    alpha = [] ; alpha_ = [] ; xp = [] ; xp_ = []
    for i in range(len(zeta)):
        alpha.append(zeta[i] ** 2.0 * a)
        alpha_.append(np.transpose(alpha[-1]))
        xp.append((2.0 * alpha[-1] / np.pi) ** 0.75)
        xp_.append(np.transpose(xp[-1]))
        
    # Numero de funciones gaussianas contraidas en la base
    n_basis = 3
    
    # Obtener la matriz de densidad del calculo scf
    P = SCF(coordinates, N, Z, d, alpha, alpha_, xp, xp_, n_basis)

    # Trazar la gráfica a lo largo del eje z
    n_points = 100
    z_points = np.linspace(-lim, lim, n_points)
    CGF_psi = [np.zeros(n_points), np.zeros(n_points)]

    for z in range(n_points):

    	# Por cada núcleo, para centrar al orbital
    	for i in range(len(Z)):

    		# Por cada gaussiana contraida
    		for j in range(n_basis):

    			r = np.array([[0, 0, z_points[z]]])
    			radial = np.linalg.norm(r - coordinates[i])
    			CGF_psi[i][z] += d[0, j] * xp_[i][0, j] * \
    							 np.exp(- alpha_[i][0, j] * radial**2.)


    # Calcular densidad en cada punto
    rho = np.zeros(n_points)

    # Por cada fila de P
    for i in range(N):

    	# Por cada columna de P
    	for j in range(N):

    		rho += P[i, j] * CGF_psi[i] * CGF_psi[j]

    # Graficar
    plt.plot(z_points, rho)
    plt.title(r'Electron density $\Psi^2$ along Z axis')
    plt.xlabel('Z axis')
    plt.ylabel(r'$\Psi^2$')
    plt.savefig('H2_electron_density.png')
    print('\n Figure saved!')

if __name__ == '__main__':

	main()
