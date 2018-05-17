##########################
# diffevol.py            #
# Differential Evolution #
#   by James V. Soukup   #
#   for CEE 290 HW #3    #
##########################
##################################################
# The models and cost functions are in models.py #
##################################################
import models 
import matplotlib.pyplot as plt
import numpy as np
from numpy import random as rnd

##################################################
# Differential Evolution                         #
#  Parameters:                                   #
#    Pop    - Initial Population                 #
#    Cost   - Costs of Initial Pop               #
#    cr     - Crossover Probability              #
#    gam    - Child variability factor           #
#    pmut   - Mutation Probability               #
#    i      - Generation Counter                 #
#    imax   - Max Generation Count               #
#    cf     - Cost Function                      #
#    cfargs - Cost Function Arguments            #
#    mf     - Model Function                     #
#    mfargs - Model Function Arguments           #
##################################################
def diffevol(Pop,cost,cr,gam,pmut,i,im,h,etol,cf,carg):
    # Check Generation Counter #
    if (im <= i):
        # Maximum Number of generations reached
        # Return the current population
        return [Pop,cost]
    #########################
    # Step One: Selection   #
    #########################
    # Generate two unique random integers #
    # for each member of the population   #
    r = rnd.choice(Pop[:,0].size, (Pop[:,0].size,2))
    # Replace pairs of duplicates with a unique pair
    dup    = r[:,0]==r[:,1]
    r[dup] = rnd.choice(Pop[:,0].size,r[dup].shape,False)
    # Define the mating partners
    FirstMates = Pop[r[:,0],:]
    SecndMates = Pop[r[:,1],:]
    ####################
    # Step Two: Mating #
    ####################
    # Partial Crossover
    Pcr = rnd.choice([0,1],Pop.shape,p=[1-cr,cr])
    # Recombination
    mateDiff = np.subtract(FirstMates,SecndMates)
    crssover = np.multiply(gam*Pcr,mateDiff)
    Child    = np.mod(np.add(Pop,crssover),1)
    # Mutation
    Mut = rnd.rand(*Child.shape)
    Mut = Mut<pmut
    Child[Mut] = rnd.rand(*Child[Mut].shape)
    #########################
    # Step Three: Rejection #
    #########################
    # Evaluate Cost for Child Population
    childCost = cf(Child,carg)
    costc = childCost[1][1]
    costp = cost[1][1]
    # Replace dominated offspring with parent
    if np.isnan(np.sum(costc)):
        # COST FUNCTION FAILURE
        print("Cost Function Failure")
        costc[~np.isnan(costc)] = 1e9
    dom = np.array(np.greater(costc,costp)).reshape((-1,))
    Child[dom] = Pop[dom]
    np.minimum(costc,costp,out=costc)
    childCost[1][1] = costc

    # Best in show
    best = np.min(costc)
    h[i] = best
    if best <= etol:
        return [Child,childCost]

    ##############################
    # Create the next generation #
    ##############################
    return diffevol(Child,childCost,cr,gam,pmut,i+1,im,h,etol,cf,carg)

##################################################
#   Differential Evolution Test#1 - Slug Model   #
##################################################
# Variable Dictionary:                           #
#   N - Population Size                          #
#   p - Number of parameters                     #
#   g - Child Variability (gamma)                # 
#   c - Crossover Probability (Pcr)              #
#   d - Observation Well distance [m]            #
#   Q - Slug volume [m3]                         #
#   m - Mutation Probability (Pmut)              #
#   h - Observation Data                         #
#   t - Time of sampling                         #
#   Pop - Initial population                     #
#     Pop[1] - S                                 #
#     Pop[2] - T                                 #
##################################################
def slugModelTest():
    print("SLUG MODEL TEST")
    # Initialize
    etol = 1.0e-6
    G = 250
    N = 20
    p = 2
    g = 0.7
    c = 0.9
    d = 10
    Q = 50
    m = float(1)/float(p)
    h = np.array([0.55,0.47,0.30,0.22,0.17,0.14])
    t = np.array([5.00,10.0,20.0,30.0,40.0,50.0])
    Pop  = rnd.rand(N,p)
    hs = np.zeros(G)
    cost = models.slugCost(Pop,[t,Q,d,h])
    hs[0] = np.min(cost[1][1])
    # Differential Evolution parameters
    # Run the Differential Evolution Algorithm
    finalOut = diffevol(Pop,cost,c,g,m,1,G,hs,etol,
                        models.slugCost,[t,Q,d,h])
    # Interpret the Output
    finalPops = finalOut[0]
    finalSims = finalOut[1]
    #finalStor = finalSims[0]
    finalCost = finalSims[1]
    finalSSRs = finalCost[1]
    opt = finalPops[np.argmin(finalSSRs)]
    # Print the results
    print("Parameter Values: {0}".format(opt))
    print("Cost: {0}".format(np.min(finalSSRs)))
    ns = np.arange(hs.size)
    plt.plot(ns,hs)
    plt.show()

def interceptionModelTest():
    print("INTERCEPTION MODEL TEST")
    ################################
    #  Initialization:             #
    #   Population Size (N)        #
    #   Number of parameters (p)   #
    #   Child Variability (gam)    #
    #   Crossover Probability (cr) #
    #   Mutation Probability (pmut)#
    ################################
    etol = 1.0e-2
    G = 250
    N = 50
    p = 4
    c = 0.9
    g = 0.7
    m = 0.25
    obs = np.genfromtxt('measurement.csv',delimiter=',')
    obsTime = obs[:,0]
    #obsStor = obs[:,1]
    dt = obsTime[1]-obsTime[0]
    Pop = rnd.uniform(0.01, 1.00,(N,p))
    h = np.zeros(G)
    cf = models.interceptCostRKF45
    cost = cf(Pop,[obs,dt])
    costc = cost[1][1]
    if np.isnan(np.sum(cost[1][1])):
        # COST FUNCTION FAILURE
        print("Cost Function Failure")
        costc[~np.isnan(costc)] = 10000
    h[0] = np.min(cost[1][1])
    print("Start DE")
    finalOut  = diffevol(Pop,cost,c,g,m,0,G,h,etol,cf,[obs,dt])
    # Interpret the Output
    finalPop  = finalOut[0]
    finalCost = finalOut[1]
    finalSSR  = finalCost[1][1]
    #finalSims = finalCost[0]
    opt = np.argmin(finalSSR)
    optPars = finalPop[opt]
    #optSims = finalSims[opt]
    #optTime = optSims[:,0]
    #optStor = optSims[:,1]
    print("Parameter Values: {0}".format(optPars))
    print("Cost: {0:8.4f}".format(np.min(finalSSR)))
    #for p in finalSims:
    #    plt.plot(p[:,0],p[:,1])
    #plt.plot(obsTime,obsStor,'bs')
    ns = np.arange(h.size)
    plt.plot(ns,h)
    plt.show()
    return

slugModelTest()
interceptionModelTest()