########################################
# CEE 290 Models                       #
#   Written by: Jasper A. Vrugt        #
#   Adapted by: James  V. Soukup       #
########################################

import numpy as np
import rungekutta as rk

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
    return [slug,cost]

#############################################
# Interception Model                        #
#   Arguments:                              #
#     t  = time [days]                      #
#     S  = storage [mm]                     #
#     T  = Observation Domain               #
#     P  = rainfall [mm/day]                #   
#     E0 = potential evaporation [mm/day]   #
#     a  = interception efficiency [-]      #
#     b  = drainage parameter [1/d]         #
#     c  = maximum storage capacity [mm]    #
#     d  = evaporation efficiency [-]       #
#   Output:                                 #
#     dS/dt = change in storage with time   #
#############################################
def interceptionModel(t,S,T,Pr,E0,a,b,c,d):
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
    Pr_int = np.interp(t,T,Pr)
    # Interpolate Potential Evap from data
    E0_int = np.interp(t,T,E0)
    # Compute Interception
    I = am*Pr_int
    # Compute Drainage dependent on storage
    # Drainage is zero by default
    D = np.zeros(am.size)
    # Theoretical drainage values
    G = bm*(S-cm)
    # Drainage when storage exceeds capacity
    D[S>cm] = G[S>cm]
    # Compute Evaporation
    cm1 = np.divide(1,cm)
    E = dm*E0_int*S*cm1
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
#   Args[0] = Observations                       #
#   Args[1] = Observation Interval               #
#  Obs                                           #
#   Obs[:,0] = Observed Times                    #
#   Obs[:,1] = Observed Storage                  #
#   Obs[:,2] = Observed Precipitation            #
#   Obs[:,3] = Observed Evaporation Potential    #
##################################################
def interceptCost(Pars, Args):
    # Extract Parameters
    a = Pars[:,0]
    b = Pars[:,1]
    c = Pars[:,2]
    d = Pars[:,3]
    # Extract Observations
    Obs = Args[0]
    dt  = Args[1]
    obsTime = Obs[:,0]
    obsStor = Obs[:,1]
    obsPrec = Obs[:,2]
    obsEvap = Obs[:,3]
    # Prepare to simulate with Runge-Kutta 4
    # Rebundle arguments
    args = [obsTime,obsPrec,obsEvap,a,b,c,d]
    # Handle for model function
    mf = interceptionModel
    # Initialize output array
    simStor = np.zeros((obsTime.size,a.size))
    # Grab initial conditiion
    simStor[0,:] = obsStor[0]
    # Iterate over observation period
    for i,t in enumerate(obsTime):
        # Index of rk4 step result
        j = i+1
        # Prevent out-of-bounds array access
        if j >= obsTime.size: break
        # Take an rk4 step
        simStor[j,:] = rk.rk4(t,simStor[i,:],dt,mf,args)
    # Return the simulation results and the cost (SSR)
    return [simStor.T,ssr(simStor.T,obsStor)]