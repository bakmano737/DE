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

######################################################################
# Differential Evolution                                             #
######################################################################
#  Recombination:                                                    #
#    child = parent + fde*(mate1-mate2)                              #
#    mate1 - First randomly selected member of the population        #
#    mate2 - Second randomly selected member of the population       #
#  Parameters:                                                       #
#    Pop    - Initial population of parameters                       #
#    cost   - Costs of initial population                            #
#    cr     - Crossover probability                                  #
#    fde    - Child variability factor                               #
#    pmut   - Mutation Probability                                   #
#    i      - Generation counter                                     #
#    im     - Max Generation Count                                   #
#    etol   - Exit Tolerance (Convergance)                           #
#    hist   - Lowest SSR of all previous generations (Analysis)      #
#    cf     - Cost Function                                          #
#    carg   - Cost Function Arguments                                #
######################################################################
def diffevol(Pop,cost,cr,fde,pmut,i,im,hist,etol,cf,carg):
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
    crssover = np.multiply(fde*Pcr,mateDiff)
    Child    = np.mod(np.add(Pop,crssover),1)
    # Mutation
    Mut = rnd.rand(*Child.shape)
    Mut = Mut<pmut
    Child[Mut] = rnd.rand(*Child[Mut].shape)
    #########################
    # Step Three: Rejection #
    #########################
    # Evaluate Cost for Child Population
    chCst = cf(Child,carg)
    costc = chCst[1][1]
    costp = cost[1][1]
    # Replace dominated offspring with parent
    dom = np.array(np.greater(costc,costp)).reshape((-1,))
    Child[dom] = Pop[dom]
    np.minimum(costc,costp,out=costc)
    chCst[1][1] = costc

    # Best in show
    best = np.min(costc)
    hist[i] = best

    # Check convergance
    #if best <= etol:
    #   return [Child,chCst]

    # Check Generation Counter 
    if (im <= i+1):
        # Maximum Number of generations reached
        # Return the current population
        return [Child,chCst]

    ##############################
    # Create the next generation #
    ##############################
    return diffevol(Child,chCst,cr,fde,pmut,i+1,im,hist,etol,cf,carg)

######################################################################
# Differential Evolution Alternate Recombination                     #
######################################################################
#  Recombination:                                                    #
#    child = parent + fde*(mate1-mate2) + lam*(best-parent)          #
#    best  - Individual with lowest SSR in current generation        #
#    mate1 - First randomly selected member of the population        #
#    mate2 - Second randomly selected member of the population       #
#  Parameters:                                                       #
#    Pop   - Initial population of parameters                        #
#    cost  - Costs of initial population                             #
#    cr    - Crossover probability                                   #
#    fde   - Child variability factor                                #
#    lam   - Best parent scaling factor                              #
#    pmut  - Mutation Probability                                    #
#    i     - Generation counter                                      #
#    im    - Max Generation Count                                    #
#    etol  - Exit Tolerance (Convergance)                            #
#    hist  - Lowest SSR of all previous generations (Analysis)       #
#    cf    - Cost Function                                           #
#    carg  - Cost Function Arguments                                 #
######################################################################
def dealt(Pop,cost,cr,fde,lam,pmut,i,im,hist,etol,cf,carg):
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
    # Best in show
    besti = np.argmin(cost)
    bestp = Pop[besti,:]
    hist[i] = cost[besti]

    ####################
    # Step Two: Mating #
    ####################
    # Partial Crossover
    Pcr = rnd.choice([0,1],Pop.shape,p=[1-cr,cr])
    # Recombination
    mateDiff = np.subtract(FirstMates,SecndMates)
    bestDiff = np.subtract(Pop,bestp)
    crssover = np.multiply(fde*Pcr,mateDiff)
    bestover = np.multiply(lam,bestDiff)
    fullover = np.add(crssover,bestover)
    Child    = np.mod(np.add(Pop,fullover),1)
    # Mutation
    Mut = rnd.rand(*Child.shape)
    Mut = Mut<pmut
    Child[Mut] = rnd.rand(*Child[Mut].shape)
    #########################
    # Step Three: Rejection #
    #########################
    # Evaluate Cost for Child Population
    chCst = cf(Child,carg)
    costc = chCst[1][1]
    costp = cost[1][1]
    # Replace dominated offspring with parent
    dom = np.array(np.greater(costc,costp)).reshape((-1,))
    Child[dom] = Pop[dom]
    np.minimum(costc,costp,out=costc)
    chCst[1][1] = costc

    # Check convergance
    #if best <= etol:
    #   return [Child,chCst]

    # Check Generation Counter 
    if (im <= i+1):
        # Maximum Number of generations reached
        # Return the current population
        return [Child,chCst]

    ##############################
    # Create the next generation #
    ##############################
    return dealt(Child,chCst,cr,fde,lam,pmut,i+1,im,hist,etol,cf,carg)

######################################################################
# Simulator Function - Use this to run DE and process the reuslts    #
######################################################################
def deSimulate(G,N,P,pcr,fde,pmut,etol,cf,carg):
    # Create the history array
    Hist = np.zeros(G)
    # Create an initial population
    Pop = rnd.rand(N,P)
    # Evaluate cost function for initial pop
    Cost = cf(Pop,carg)
    # Run DE
    FinalGen = diffevol(Pop,Cost,pcr,fde,pmut,0,G,Hist,etol,cf,carg)
    # Parse the output [Population,[[simtim,simslug],[res,ssr]]]
    FinalPop = FinalGen[0]
    FinalCst = FinalGen[1]
    FinalSSR = FinalCst[1][1]
    # Determine the individual with the lowest SSR
    optimum  = np.argmin(FinalSSR)
    # Get the parameters, cost, and simulation of the champion
    BestPars = FinalPop[optimum]
    BestCost = FinalSSR[optimum]
    BestVals = FinalCst[0][optimum]
    # Save the current output for later
    return [BestPars,BestCost,BestVals,Hist]

def deaSimulate(G,N,P,pcr,fde,lam,pmut,etol,cf,carg):
    # Create the history array
    Hist = np.zeros(G)
    # Create an initial population
    Pop = rnd.rand(N,P)
    # Evaluate cost function for initial pop
    Cost = cf(Pop,carg)
    # Run DE
    FinalGen = dealt(Pop,Cost,pcr,fde,lam,pmut,0,G,Hist,etol,cf,carg)
    # Parse the output [Population,[[simtim,simslug],[res,ssr]]]
    FinalPop = FinalGen[0]
    FinalCst = FinalGen[1]
    FinalSSR = FinalCst[1][1]
    # Determine the individual with the lowest SSR
    optimum  = np.argmin(FinalSSR)
    # Get the parameters, cost, and simulation of the champion
    BestPars = FinalPop[optimum]
    BestCost = FinalSSR[optimum]
    BestVals = FinalCst[0][optimum]
    # Save the current output for later
    return [BestPars,BestCost,BestVals,Hist]

# Problem #8 - Plots
# Use this function to plot the reuslts for Slug and Storage models
def dePlots():
    # Number of Generations
    G = 250
    # Population Size
    N = 50
    # Other DE Parameters
    # Crossover Probability
    pcr = 0.9
    # Recombination Variability
    fde = 0.7
    # Recombination Variability (Exploration)
    fdea = 0.3
    # Exploitation Factor
    lam = 0.7
    # Exit Error Tolerance
    etol = 1e-6

    ##################################################################
    # Simulation of the Models
    ##################################################################
    # Slug Model Initialization
    # Observed Slug data
    obsSlug  = np.array([0.55,0.47,0.30,0.22,0.17,0.14])
    obsTime  = np.array([5.00,10.0,20.0,30.0,40.0,50.0])
    # Slug model known values
    sP = 2 # Number of paramters used by slug model
    d = 10 # Observation well distance (m)
    Q = 50 # Injected slug colume (m^3)
    spm = float(1)/float(sP) # Slug model mutation probability
    # Slug model cost function handle and arguments
    scf = models.slugCost 
    sca = [obsTime,Q,d,obsSlug] # Slug model cost function args
    Slugs = [] # Initial empty array for results
    Sluga = [] # Initial empty array for alternate results
    
    # Interception Model Initialization
    # Observed Storage, Precipitation, and Potential Evaporation
    Obs = np.genfromtxt('measurement.csv',delimiter=',')
    # Observation Time Step
    dt = Obs[1,0] - Obs[0,0]
    iP = 4 # Number of parameters used by Interception Model
    ipm = float(1)/float(iP) # Interception model mutation probability
    # Interception Model Cost Function Handle
    icf = models.interceptionModel_CF
    # Interception Model args
    ica = [Obs,dt]
    Stors = [] # Initial empty array for storage results
    Stora = [] # Initial empty array for alternate storage results

    # Run Simulations
    sims = 3 # Number of simulatins to run
    while sims > 0:
        Slugs.append(deSimulate( G,N,sP,pcr,fde,     spm,etol,scf,sca))
        Sluga.append(deaSimulate(G,N,sP,pcr,fdea,lam,spm,etol,scf,sca))
        Stors.append(deSimulate( G,N,iP,pcr,fde,     ipm,etol,icf,ica))
        Stora.append(deaSimulate(G,N,iP,pcr,fdea,lam,ipm,etol,icf,ica))
        sims -= 1

    ##################################################################
    # Print and Plot Simulation Results
    ##################################################################
    slugPlt = plt.subplot(221)
    storPlt = plt.subplot(222)
    alugPlt = plt.subplot(223)
    atorPlt = plt.subplot(224)
    for i,(slug,stor,alug,ator) in enumerate(zip(Slugs,Stors,Sluga,Stora)):
        simn = "Sim {0}".format(i+1)
        print(simn)
        print("Slug Model")
        print("\tParameter Values:")
        print("\t\tS={0:6.4f}".format(slug[0][0]))
        print("\t\tT={0:6.4f}".format(slug[0][1]))
        print("\tCost: {0:10.6f}".format(np.min(slug[1])))
        slugPlt.semilogy(slug[3], label=simn)
        print("Interception Model")
        print("\tParameter Values:")
        print("\t\ta={0:6.4f}".format(stor[0][0]))
        print("\t\tb={0:6.3f}".format(999*stor[0][1]+1))
        print("\t\tc={0:6.4f}".format(5.0*stor[0][2]))
        print("\t\td={0:6.4f}".format(3.0*stor[0][3]))
        print("\tCost: {0:10.6f}".format(np.min(stor[1])))
        storPlt.semilogy(stor[3], label=simn)
        print("Slug Model: Alternate Recombination")
        print("\tParameter Values:")
        print("\t\tS={0:6.4f}".format(alug[0][0]))
        print("\t\tT={0:6.4f}".format(alug[0][1]))
        print("\tCost: {0:10.6f}".format(np.min(alug[1])))
        alugPlt.semilogy(alug[3], label=simn)
        print("Interception Model: Alternate Recombination")
        print("\tParameter Values:")
        print("\t\ta={0:6.4f}".format(ator[0][0]))
        print("\t\tb={0:6.3f}".format(999*ator[0][1]+1))
        print("\t\tc={0:6.4f}".format(5.0*ator[0][2]))
        print("\t\td={0:6.4f}".format(3.0*ator[0][3]))
        print("\tCost: {0:10.6f}".format(np.min(ator[1])))
        atorPlt.semilogy(ator[3], label=simn)
    slugPlt.set_xlabel('Generation')
    slugPlt.set_ylabel('Minimum SSR')
    slugPlt.set_title('Slug Model')
    slugPlt.legend()
    storPlt.set_xlabel('Generation')
    storPlt.set_ylabel('Minimum SSR')
    storPlt.set_title('Interception Model')
    storPlt.legend()
    alugPlt.set_xlabel('Generation')
    alugPlt.set_ylabel('Minimum SSR')
    alugPlt.set_title('Slug Model: Alternate Recombination')
    alugPlt.legend()
    atorPlt.set_xlabel('Generation')
    atorPlt.set_ylabel('Minimum SSR')
    atorPlt.set_title('Interception Model: Alternate Recombination')
    atorPlt.legend()
    plt.show()

dePlots()