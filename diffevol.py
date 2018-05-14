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
def diffevol(Pop,cost,cr,gam,pmut,i,imax,cf,carg):
    # Check Generation Counter #
    if (imax < i):
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
    #print Mut.shape
    Child[Mut] = rnd.rand(*Child[Mut].shape)
    #print Child.shape
    #########################
    # Step Three: Rejection #
    #########################
    # Evaluate Cost for Child Population
    childCost = cf(Child,carg)
    costc = childCost[1]
    costp = cost[1]
    # Replace dominated offspring with parent
    dom = np.array((costc>costp)).reshape((-1,))
    Child[dom] = Pop[dom]
    costc = np.minimum(costc,costp)
    childCost[1] = costc

    ##############################
    # Create the next generation #
    ##############################
    diffevol(Child,costc,cr,gam,pmut,i+1,imax,cf,carg)
    return [Child,childCost]

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
    N = 200
    p = 2
    g = 0.7
    c = 0.9
    d = 10
    Q = 50
    m = float(1)/float(p)
    h = np.array([0.55,0.47,0.30,0.22,0.17,0.14])
    t = np.array([5.00,10.0,20.0,30.0,40.0,50.0])
    Pop  = rnd.rand(N,p)
    cost = models.slugCost(Pop,[t,Q,d,h])
    # Differential Evolution parameters
    # Run the Differential Evolution Algorithm
    finalOut = diffevol(Pop,cost,c,g,m,0,900,
                        models.slugCost,[t,Q,d,h])
    # Interpret the Output
    finalPop  = finalOut[0]
    finalCost = finalOut[1]
    finalSSR  = finalCost[1]
    opt = finalPop[np.argmin(finalSSR)]
    # Print the results
    print("Parameter Values: {0}".format(opt))
    print("Cost: {0}".format(np.min(finalSSR)))

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
    obs = np.genfromtxt('measurement.csv',delimiter=',')
    obsTime = obs[:,0]
    dt = obsTime[1]-obsTime[0]
    Pop = rnd.rand(200,4)
    cost = models.interceptCost(dt,obs,Pop)
    finalSSR  = cost[1]
    opt = Pop[np.argmin(finalSSR)]
    print("Parameter Values: {0}".format(opt))
    print("Cost: {0}".format(np.min(finalSSR)))
    return cost

slugModelTest()
interceptionModelTest()