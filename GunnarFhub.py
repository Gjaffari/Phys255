import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.special import binom
from time import time

def findConnect(nsites):
    if nsites == 6:
        connect = [(0,1), (0,3),
                   (1,2), (1,4),
                   (2,0), (2,5),
                   (3,4), (3,0),
                   (4,5), (4,1),
                   (5,3), (5,2)]
    
    
    elif nsites == 8:
         connect = [(0,1), (0,3),
                    (1,6), (1,4),
                    (2,3), (2,1),
                    (3,4), (3,6),
                    (4,5), (4,7),
                    (5,2), (5,0),
                    (6,7), (6,5),
                    (7,0), (7,2)]
    elif nsites == 10:
         connect = [(0,1), (0,3),
                    (1,2), (1,4),
                    (2,3), (2,5),
                    (3,4), (3,6),
                    (4,5), (4,7),
                    (5,6), (5,8),
                    (6,7), (6,9),
                    (7,8), (7,0),
                    (8,9), (8,1),
                    (9,0), (9,2)]
    else:
        if type(nsites) != int and type(nsites) != float:
            raise TypeError('findConnect()\nNumber of sites can only be integer or float not ' + str(type(nsites)).strip('<').strip('>').strip('class ').strip("'") + '.')
        else:
            raise ValueError('findConnect()\nNumber of sites can only be 6,8, or 10 not %d.' % nsites)
    return connect

def isKthBitSet(n,k):
    if n & (1 << k):
        return 1
    else:
        return 0

def find_state(state, configs, N, M):
    for n in range(M):
        for k in range(2*N):
            if state[k] != configs[n,k]:
                break
            else:
                return n
    raise ValueError('find_state()\nState not found')

def FHub(U, N, t = 1., T = np.logspace(-2,2,51), mu = np.linspace(-10,10,31)):
    time0 = time()
    NT = len(T)
    Nmu = len(mu)

    Z    = np.zeros((Nmu,NT))
    E    = np.zeros((Nmu,NT))
    E2   = np.zeros((Nmu,NT))
    Erho = np.zeros((Nmu,NT))
    #Mag  = np.zeros((Nmu,NT))
    #Mag2 = np.zeros((Nmu,NT))
    Saf  = np.zeros((Nmu,NT))
    rho  = np.zeros((Nmu,NT))
    rho2 = np.zeros((Nmu,NT))
    D    = np.zeros((Nmu,NT))
    m2   = np.zeros((Nmu,NT))

    Mtot = 0
    w = 1
    for Ne in range(2*N+1):
        for totSz in range(-Ne, Ne+2, 2):
            time1 = time()
            addon = ' seconds.'
            Nup = (Ne + totSz) / 2
            Ndn = (Ne - totSz) / 2
            if Nup > N or Ndn > N:
                continue
            else:
                M = int(binom(N,Nup) * binom(N,Ndn))
                Mtot += M

                part  = np.zeros((M,2*N),int)
                confg = np.zeros(2*N, int)
                #Safj  = np.zeros(M)
                Dj    = np.zeros(M)
                m2j   = np.zeros(M)
############################Construct States#######################################
                l=0
                for n in range(4**N):
                    Sz_sum = 0
                    Ne_sum = 0
                    for i in range(2*N):
                        confg[i] = isKthBitSet(n,i)
                        if i%2 == 0:
                            Sz_sum += confg[i]
                        elif i%2 == 1:
                            Sz_sum -= confg[i]
                        Ne_sum += confg[i]
                    if Sz_sum == totSz and Ne_sum == Ne:
                        part[l,:] = confg[:]
                        for i in range(N):
                            iup = 2*i
                            idn = iup+1
                            m2j[l] += (part[l,iup]-part[l,idn])**2
                            Dj[l]  += part[l,iup]*part[l,idn]
                            for j in range(N):
                                jup = 2*j
                                jdn = jup+1
                                #Safj[l] += ((part[l,iup]-part[l,idn])*(part[l,jup]-part[l,jdn]))
                        l += 1
                if l != M:
                    raise ValueError('FHub()\nl not equal to M. l=%d, M=%d, Ne%d, Sz=%d' % (l,M,Ne,totSz))
                H = np.zeros([M,M])
                for n in range(M):
                    for i in range(N):
###############################Diagonal Terms######################################
                        H[n,n] = U*Dj[n] - Ne*U/2. + U/4.
#############################Off-Diagonal Terms####################################
                        for j,s in findConnect(N):
                            new_state = np.copy(part[n,:])
                            updns = [(2*j,2*s), (2*j+1, 2*s+1)]
                            for g,h in updns:
                                if part[n,g] == 1 and part[n,h] == 0:
                                    new_state[g] = 0
                                    new_state[h] = 1
                                    m = find_state(new_state,part,N,M)
                                    H[n,m] -= t
                                elif part[n,g] == 0 and part[n,h] == 1:
                                    new_state[g] = 1
                                    new_state[h] = 0
                                    m = find_state(new_state,part,N,M)
                                    H[n,m] -= t
                vals, vecs = np.linalg.eigh(H)
#########################Thermal Averages#########################################
                #Safa = np.zeros(M)
                Da   = np.zeros(M)
                m2a  = np.zeros(M)
                for n in range(M):
                    #Safa[n] = np.sum(vecs[:,n]**2*Safj[:])
                    Da[n]   = np.sum(vecs[:,n]**2*Dj[:])
                    m2a[n]  = np.sum(vecs[:,n]**2*m2j[:])
                for imu in range(Nmu):
                    for iT in range(NT):
                        boltz = np.exp(-(vals[:]-Ne*mu[imu])/T[iT])
                        Z[   imu, iT] += np.sum(             boltz)
                        rho[ imu, iT] += np.sum(Ne*          boltz)
                        rho2[imu, iT] += np.sum(Ne**2*       boltz)
                        E[   imu, iT] += np.sum((vals[:])*   boltz)
                        E2[  imu, iT] += np.sum((vals[:])**2*boltz)
                        Erho[imu, iT] += np.sum((vals[:])*Ne*boltz)
                        #Mag[ imu, iT] += np.sum(totSz*       boltz)
                        #Mag2[imu, iT] += np.sum(totSz**2*    boltz)
                        #Saf[ imu, iT] += np.sum(Safa[:]*     boltz)
                        D[   imu, iT] += np.sum(Da[:]*       boltz)
                        m2[  imu, iT] += np.sum(m2a[:]*      boltz)

                mytime = time() - time1
                if mytime > 3600:
                    mytime /= 3600.
                    addon = ' hours.'
                elif mytime > 60:
                    mytime /= 60.
                    addon = ' minutes.'
                mytime = str(round(mytime,2))
                print('Sector %d of %d completed in ' % (w, (N+1)**2) + mytime + addon)
                w += 1
    E     /= Z
    E2    /= Z
    Erho  /= Z
    #Mag   /= Z
    #Mag2  /= Z
    #Saf   /= Z
    rho   /= Z
    rho2  /= Z
    D     /= Z
    m2    /= Z
    Cv = (E2 - (Erho - (E * rho))**2 / rho2) / T**2 
    
    np.savetxt('m2_N%d_U%d.dat'   % (N,U), m2)
    np.savetxt('D_N%d_U%d.dat'    % (N,U), D)
    np.savetxt('rho_N%d_U%d.dat'  % (N,U), rho)
    np.savetxt('E_N%d_U%d.dat'    % (N,U), E)
    #np.savetxt('Saf_N%d_U%d.dat'  % (N,U), Saf)
    np.savetxt('Cv_N%d_U%d.dat'   % (N,U), Cv)
    #np.savetxt('Temps.dat', T)

    if time()-time0 > 3600:
        newtime = str(round((time()-time0)/3600.,2))
        addon = ' hours.'
    elif time()-time0 > 60:
        newtime = str(round((time()-time0)/60.,2))
        addon = ' minutes.'
    else:
        newtime = str(round(time()-time0,2))
        addon = ' seconds.'
    print('The whole process took ' + newtime + addon)
    return
