########################################
# CEE 290 Models                       #
#   Written by: Jasper A. Vrugt        #
#   Adapted by: James  V. Soukup       #
########################################

import numpy as np

########################################
# Sum of Squared Residuals             #
#  (Cost Function)                     #
#  Parameters:                         #
#    p - Model Function Parameters     #
#    mf - Model Function               #
#    mfargs - Model Functin Arguments  #
#    d - Data to compare to the model  #
#  Output:                             #
#    (d-f(p))'*(d-f(p))                #
########################################
def ssr(p,d):
    res = p - d
    ssr = np.sum(np.multiply(res,res),axis=1)
    return [res,ssr]

########################################
# Runge Kutta Fourth Order: Integrator #
########################################
# Input Dictionary                     #
#  t0 - Current Simulation Time        #
#  y0 - Current Simulation Value       #
#  dt - Time step to next simulation   #
#   f - Model Function                 #
#  pars - Model Parameters             #
#  args - Model Function Arguments     #
########################################
# This Integrator expects a particular #
# model function type. The model must  #
# represent a differential equation of #
# the form: y' = f(t,y)                #
# Therefore the function must take as  #
# input the current simulation value,  #
# the current simulation time, and any #
# additional inputs. It must output y' #
# for the given time and estimate      #
########################################
def rk4(t0,y0,dt,f,fargs):
    s1 = f(t0,     y0,     *fargs)
    s2 = f(t0+dt/2,y0+s1/2,*fargs)
    s3 = f(t0+dt/2,y0+s2/2,*fargs)
    s4 = f(t0+dt,  y0+s3,  *fargs)
    return y0+dt*(s1+2*s2+2*s3+s4)/6
    
########################################
# Slug Model                           #   
#  Arguments:                          #
#    pars - Paramters [S,T]            #
#       t - Time                       #
#       Q - Amount of injected water   #
#       d - Distance from injection    #
########################################
def slugModel(pars,t,Q,d):
    t = np.matrix(t)
    S = np.matrix(pars[:,0]).T
    T = np.matrix(pars[:,1]).T
    # Predict "h" using the slug model
    # Separated into four lines for clarity
    A = Q/(4 * np.pi * T * t)
    expP = np.divide(-(d**2)*S,4*T)
    expt = 1/t
    return np.multiply(A,np.exp(expP*expt))

########################################
# Slug Model Cost:                     #
#  Returns the cost (residuals, ssr)   #
#  of the Slug Model                   #
#  Slug Model is run by the given Pars #
########################################
# Parameter                            #
#  Pars                                #
#   Pars[0] - S                        #
#   Pars[1] - T                        #
#  Args                                #
#   Args[0] - t                        #
#   Args[1] - Q                        #
#   Args[2] - d                        #
#   Args[3] - Data                     #
########################################
def slugCost(Pars, Args):
    t = Args[0]
    Q = Args[1]
    d = Args[2]
    data = Args[3]
    slug = slugModel(Pars,t,Q,d)
    cost = ssr(slug,data)
    return cost

#############################################
# Interception Model                        #
#   Arguments:                              #
#     t  = time [days]                      #
#     S  = storage [mm]                     #
#     P  = rainfall [mm/day]                #   
#     E0 = potential evaporation [mm/day]   #
#     a  = interception efficiency [-]      #
#     b  = drainage parameter [1/d]         #
#     c  = maximum storage capacity [mm]    #
#     d  = evaporation efficiency [-]       #
#   Output:                                 #
#     dS/dt = change in storage with time   #
#############################################
    # Map parameters to feasible space
    # a - 0-1    needs no mapping
    # b - 1-1000 bm = 999*b+1
    # c - 0-5    cm = 5*c
    # d - 0-3    dm = 3*d
    am = a
    bm = 999*b+1
    cm = 5*c
    dm = 3*d
    # Interpolate Precipitation from data
    Pr_int = np.max(np.interp(t,T,Pr),0)
    # Interpolate Potential Evap from data
    E0_int = np.max(np.interp(t,T,E0),0)
    # Compute Interception
    I = a*Pr_int
    # Compute Drainage dependent on storage
    # Drainage is zero by default
    D = np.zeros(a.size)
    # Theoretical drainage values
    G = b*(S-c)
    # Drainage when storage exceeds capacity
    D[S>c] = G[S>c]
    # Compute Evaporation
    E = d*E0_int*S/c
    # Return change in storage
    return I-D-E

##################################################
# Interception Model Cost                        #
#  Returns the cost (residuals, ssr)             #
#  of the Interception Model run with given Pars #
##################################################
# Parameters:                                    #
#  Pars                                          #
#   Pars[0] = a                                  #
#   Pars[1] = b                                  #
#   Pars[2] = c                                  #
#   Pars[3] = d                                  #
#  Args                                          #
#   Args[0] = t                                  #
#   Args[1] = S                                  #
#   Args[2] = P                                  #
#   Args[3] = E0                                 #
#   Args[4] = Data                               #
##################################################
def interceptCost(dt, Args, Pars):
    a = Pars[:,0]
    b = Pars[:,1]
    c = Pars[:,2]
    d = Pars[:,3]
    obsTime = Args[:,0]
    obsStor = Args[:,1]
    obsPrec = Args[:,2]
    obsEvap = Args[:,3]
    simStor = np.zeros((obsStor.size,a.size))
    args = [obsTime,obsPrec,obsEvap,a,b,c,d]
    for i,t in enumerate(obsTime):
        simStor[i,:] = rk4(t,obsStor[0],dt,interceptionModel,args)
    return ssr(simStor.T,obsStor)