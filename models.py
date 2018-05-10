##################################
# CEE 290 Models                 #
#   Written by: Jasper A. Vrugt  #
#   Adapted by: James  V. Soukup #
##################################

import numpy as np

######################################
# Slug Model                         #   
#  Arguments:                        #
#    pars - Paramters [S,T]          #
#       t - Time                     #
#       Q - Amount of injected water #
#       d - Distance from injection  #
######################################
def slugmodel(pars,t,Q,d):
    # Define the storage and transmissivity
    t = np.matrix(t)
    S = np.matrix(pars[:,0]).T
    T = np.matrix(pars[:,1]).T

    # Predict "h" using the slug model equation (see reader)
    # Separated into four lines for clarity
    A = Q/(4 * np.pi * T * t)
    expP = np.divide(d**2*S,4*T)
    expt = 1/t
    return np.multiply(A,np.exp(expP*expt))

############################################
# Interception Model                       #
#   Arguments:                             #
#     t  = time [days]                     #
#     S  = storage [mm]                    #
#     P  = rainfall [mm/day]               #   
#     E0 = potential evaporation [[mm/day] #
#     a  = interception efficiency [-]     #
#     b  = drainage parameter [1/d]        #
#     c  = maximum storage capacity [mm]   #
#     d  = evaporation efficiency [-]      #
#   Output:                                #
#     dS/dt = change in storage with time  #
############################################
def interceptionModel(a,b,c,d,flag,P,E0,t,S):
    # Note: P and E0 are vectors with rainfall and potential evapotranspiration
    # They are thus time varying. Each iteration we thus need to derive their
    # values from interpolation. At current time, rainfall is equal to:
    # First column of P is time, second is rainfall 
    P_int = np.max(np.interp(t,P[:,1],P[:,2]),0); 
    # Same for E0
    E0_int = np.max(np.interp(t,E0[:,1],E0[:,2]),0); 
    # Calculate interception (unknown parameter x rainfall intensity)
    I = a * P_int;              # --> in mm/day
    # Calculate drainage (only if storage larger than storage capacity)
    if S > c:
        D = b*(S - c)
    else:
        D = 0
    # Calculate evaporation
    E = d * E0_int * S / c
    # Now calculate the change in storage 
    return I - D - E 