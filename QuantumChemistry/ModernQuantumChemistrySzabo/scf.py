'''
Autor: Alberto Cabrera
Contacto: albertocabja@gmail.com
Realiza el calculo scf (restringido) con la base STO-3G
'''

# Librerias para el calculo SCF
import numpy as np
from scipy.special import erf

# Funciones matematicas para SCF***********************************************
def F0(M):
    '''
    Funcion de Boys para orbital 1s, con matriz como argumento
    '''
    
    Q = np.zeros(M.shape)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] < 1e-6:
                Q[i, j] += 1.0 - M[i, j]/3.0
            else:
                Q[i, j] += 0.5 * (np.pi/M[i, j])**0.5 * erf(M[i, j]**0.5)
                
    
    return Q

def AxB(a, a_, ri, rj, n_basis):
    '''
    Vector centro del producto entre dos gaussianas, ordenados en columna
    '''
    
    Q = np.zeros([n_basis*n_basis, 3])
    count = 0
    for i in range(n_basis):
        for j in range(n_basis):
            Q[count] += a_[0, j]*rj + a[i, 0]*ri
            count += 1
            
    Q /= np.reshape(a + a_, (n_basis*n_basis, 1))
            
    return Q

def reorganizar(M, n_basis):
    '''
    Matriz cuadrada que contiene la norma de los vectores de M
    '''
    
    Q = np.zeros([n_basis*n_basis])
    for i in range(M.shape[0]):
        Q[i] += np.linalg.norm(M[i])
        
    return np.reshape(Q, (n_basis, n_basis))

def ee_integral(nb, ai, aj, ak, al, xpi, xpj, xpk, xpl, d, Ri, Rj, Rk, Rl):
    '''
    Integral de dos electrones (ij|kl)
    '''
    
    Vee = 0
    
    for m in range(nb):
        for n in range(nb):
            for o in range(nb):
                for p in range(nb):
                    
                    # Vectores de los centros en (ij|kl)
                    Rbra = np.linalg.norm(Ri - Rj)
                    Rket = np.linalg.norm(Rk - Rl)
                    Rp = (ai[0, m]*Ri + aj[0, n]*Rj)/(ai[0, m] + aj[0, n])
                    Rq = (ak[0, o]*Rk + al[0, p]*Rl)/(ak[0, o] + al[0, p])
                    Rr = np.linalg.norm(Rp - Rq)
                    
                    # Constante de normalizacion
                    norm = xpi[0, m]*xpj[0, n]*xpk[0, o]*xpl[0, p] * \
                            d[0, m]*d[0, n]*d[0, o]*d[0, p]
                    
                    # Primer termino
                    ee1 = 2.0*np.pi**(2.5)/((ai[0, m]+aj[0, n])*(ak[0, o]+ \
                                      al[0, p])*(ai[0, m]+aj[0, n]+ak[0, o]+ \
                                        al[0, p])**(0.5))
                    
                    # Segundo termino
                    ee2 = np.exp(-ai[0, m]*aj[0, n]*Rbra**2.0/(ai[0, m] + \
                                 aj[0, n]) - ak[0, o]*al[0, p]*Rket**2.0 / \
                                    (ak[0, o] + al[0, p]))
                                 
                    # Tercer termino
                    ee3 = (ai[0, m]+aj[0, n])*(ak[0, o]+al[0, p])*Rr**2.0 / \
                            (ai[0, m] + aj[0, n] + ak[0, o] + al[0, p])
                    
                    # Todos los terminos
                    Vee += norm * ee1 * ee2 * float( F0(np.array([[ee3]])) )
                    
    return Vee
# *****************************************************************************

def main(coordinates, N, Z, d, alpha, alpha_, xp, xp_, n_basis):
    
    # *************************************************************************
    # Procedimiento SCF *******************************************************
    # *************************************************************************
    
    # Inicializar matrices de traslape S, energia cinetica T, potencial de 
    # atraccion V y Hamiltoniano-core Hcore
    S = np.zeros([N, N])
    T = np.zeros([N, N])
    V = []
    for i in range(len(Z)):
        V.append(np.zeros([N, N]))
    
    # Filas de las matrices
    for i in range(N):
        
        # Columnas de las matrices
        for j in range(N):
            
            # Constante de normalizacion para las integrales monoelectronicas
            mono_norm = xp[i] * np.transpose(xp[j]) * d * np.transpose(d)
            
            # Vector entre orbitales. Aprovechando que hay un orbital
            # por atomo, se utilizan las coordenadas nucleares
            R = np.linalg.norm(coordinates[i] - coordinates[j])
            
            # Matrices que contienen las operaciones aritmeticas 
            # entre alphas, necesarias para calcular las integrales
            sum_ = alpha[i] + alpha_[j]
            mul_ = alpha[i] * alpha_[j]
            
            # Matriz que contiene el centro del producto de gaussianas, 
            # ordenada en una sola columna de vectores
            RP = AxB(alpha[i],alpha_[j],coordinates[i],coordinates[j],n_basis)
            
            # Integral de traslape (i|j)
            overlap = (np.pi/sum_)**1.5 * np.exp(- mul_*R**2.0 / sum_)
            S[i, j] += np.sum(mono_norm * overlap)
            
            # Integral cinetica (i|-1/2*LAP^2|j)
            t1 = (mul_/sum_)*(3.0 - 2.0*R**2.0*mul_/sum_)*(np.pi/sum_)**1.5
            t2 = np.exp(- mul_ * R**2.0 / sum_)
            kinetic = t1*t2
            T[i, j] += np.sum(mono_norm * kinetic)
            
            # Por cada interaccion electron-nucleo
            for n in range(len(Z)):
                
                # Vector norma entre orbital y nucleo
                RZ = reorganizar(RP - coordinates[n], n_basis)
                
                # Integral de atraccion nuclear (i|-Z_n/r_n|j)
                n1 = -2.0*np.pi/(sum_) * np.exp(-mul_ * R**2.0 / sum_)
                n2 = F0(sum_ * RZ**2.0)
                nuclear = Z[n] * n1 * n2
                V[n][i, j] += np.sum(mono_norm * nuclear)
           
    # Hcore = T + V
    Hcore = T 
    for i in range(len(V)):    
        Hcore += V[i]
    
    # Integrales de dos electrones (ij|kl)
    TT = np.zeros([N, N, N, N])
    for i in range(N):
        for j in range(N):    
            for k in range(N):        
                for l in range(N):
                    
                    # Vectores de los centros
                    Ri = coordinates[i] ; Rj = coordinates[j]
                    Rk = coordinates[k] ; Rl = coordinates[l]
                    
                    # Integral para los respectivos i, j, k, l
                    TT[i, j, k, l] += ee_integral(n_basis, alpha_[i], 
                      alpha_[j], alpha_[k], alpha_[l], xp_[i], xp_[j],
                      xp_[k], xp_[l], d, Ri, Rj, Rk, Rl)
                    
    # Matriz de transformacion calculada con 
    # la ortogonalizacion canonica:     X = U s**-0.5
    eigS, U = np.linalg.eig(S)
    s_sqrt = eigS**-0.5 * np.identity(N)
    X = np.dot(U, s_sqrt)
    
    print('\n STARTING SCF...')
    # Guess inicial F = Hcore + 0
    F = Hcore
    P = np.zeros([N, N])
    
    # Num de iteracion
    its = 0
    
    # Num maximo de iteraciones
    max_its = 50
    
    # Valor limite para convergencia
    threshold = 1e-07
    
    while its < max_its:
        
        its += 1
        print(' CYCLE Nr. {0} \n **********************'.format(its))
        
        # F' = X_dagger F X
        Fprime = np.matmul(X.T, np.matmul(F, X))
        
        # Diagonalizar F'C' = C'e
        Cprime = np.linalg.eig(Fprime)[1]
        
        # C = X C'
        C = np.matmul(X, Cprime)
        
        # Calcular matriz de densidad P
        OldP = P
        P = np.zeros([N, N])
        for i in range(N):
            for j in range(N):
                for a in range(int(N/2)):
                    P[i, j] += C[i, a]*C[j, a]
                P[i, j] *= 2.0
        print(' DENSITY MATRIX P \n {0}'.format(P))
        
        # A partir de P calculada, obtener nueva guess en G
        G = np.zeros([N, N])
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        G[i, j] += P[k, l]*(TT[i, j, l, k] - \
                                             0.5*TT[i, k, l, j])
        
        # Calcular nueva matriz de Fock F
        F = Hcore + G
        
        # La nueva energia electronica esta dada por P y F
        E0 = np.sum( np.transpose(P) * (Hcore + F) ) * 0.5
        
        # La energia E0 debe obtenerse con la misma densidad P que la 
        # utilizada para construir F. Cuando esto ocurre, se ha alcanzado
        # la convergencia
        Delta = (np.sum((P - OldP)**2.0)/np.sum(P.shape))**0.5
        print(' E0= {0:.8f}    Delta= {1:.8f}'.format(E0,  Delta))
        if Delta < threshold:
            print(' SCF COMPLETED AFTER {0} CYCLES'.format(its))
            break
    
    
    # *************************************************************************
    # El procedimiento SCF ha terminado ***************************************
    # *************************************************************************
                    
    # La energia de repulsion nucleo-nucleo
    ENuc = 0.
    for i in range(len(Z)):
        for j in range(len(Z)):
            if j > i:
                Rnuc = np.linalg.norm(coordinates[i] - coordinates[j])
                ENuc += max([float(Z[i]), float(Z[j])]) / Rnuc
                
    # La energia total esta dada por la energia de repulsion mas la 
    # energia electronica
    Etot = E0 + ENuc
    print('\n Etot= {0:.8f}'.format(Etot))
    
    return P
    
    
