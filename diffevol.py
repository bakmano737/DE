##########################
# diffevol.py            #
# Differential Evolution #
#   by James V. Soukup   #
#   for CEE 290 HW #3    #
##########################
MAXGENS = 2000
################################
# The models live in models.py #
################################
import models 
import numpy as np
from numpy import random as rnd

#######################################
# Sum of Squared Residuals            #
#  (Cost Function)                    #
#  Parameters:                        #
#    p - Model Function Parameters    #
#    mf - Model Function              #
#    mfargs - Model Functin Arguments #
#    d - Data to compare to the model #
#  Output:                            #
#    (d-f(p))'*(d-f(p))               #
#######################################
def ssr(p,d,mf,mfargs):
    pre = mf(p,*mfargs)
    res = pre - d
    ssr = np.sum(np.multiply(res,res),axis=1)
    return ssr

###############################
# This is where I'd put other #
# cost functions              #
# ... if I had any...         #
###############################

########################################
# diffevol function                    #
#  Parameters:                         #
#    Pop    - Initial Population       #
#    cr     - Crossover Probability    #
#    gam    - Child variability factor #
#    pmut   - Mutation Probability     #
#    i      - Iteration Counter        #
#    cf     - Cost Function            #
#    cfargs - Cost Function Arguments  #
#    mf     - Model Function           #
#    mfargs - Model Function Arguments #
########################################
def diffevol(Pop, cr, gam, pmut, i, cf, cfargs, mf, mfargs):
    # Check Generation Counter #
    if (MAXGENS < i):
        # Maximum Number of generations reached
        # Return the current population

        return Pop
    #########################
    # Step One: Selection   #
    #########################
    # Generate two unique random integers #
    # for each member of the population   #
    r = rnd.choice(Pop[:,0].size, (Pop[:,0].size,2))
    # Replace any pairs of duplicates with a pair of unique labels #
    r[r[:,0]==r[:,1]] = rnd.choice(Pop[:,0].size,r[r[:,0]==r[:,1]].shape,False)
    # Define the mating partners #
    FirstMates = Pop[r[:,0],:]
    SecndMates = Pop[r[:,1],:]
    #########################
    # Step Two: Mating      #
    #########################
    # Partial Crossover #
    Pcr = rnd.choice([0,1],Pop.shape,p=[1-cr,cr])
    # Recombination #
    mateDiff = np.subtract(FirstMates,SecndMates)
    crssover = np.multiply(gam*Pcr,mateDiff)
    Child = np.mod(np.add(Pop,crssover),1)
    # Mutation #
    Mut = rnd.rand(*Child.shape)
    Mut = Mut<pmut
    #print Mut.shape
    Child[Mut] = rnd.rand(*Child[Mut].shape)
    #print Child.shape
    #########################
    # Step Three: Rejection #
    #########################
    # Extract the Cost Function and arguments
    # Evaluate Cost for Initial Population
    costi = cf(Pop,cfargs,mf,mfargs)
    #print costi.shape
    # Evaluate Cost for Child Population
    costc = cf(Child,cfargs,mf,mfargs)
    # Replace dominated offspring with parent
    dom = np.array((costc>costi)).reshape((-1,))
    Child[dom] = Pop[dom]

    ##############################
    # Create the next generation #
    ##############################
    diffevol(Child,cr,gam,pmut,i+1,cf,cfargs,mf,mfargs)
    return Child

def testFunction():
    ################################
    #  Parameters                  #
    #   Population Size (N)        #
    #   Number of parameters (p)   #
    #   Child Variability (gam)    #
    #   Crossover Probability (cr) #
    ################################
    N = 100
    p = 2
    gam = 0.7
    cr = 0.9
    pmut = float(1)/float(p)
    # Initial Population
    Pop = rnd.rand(N,p)
    h = np.array([0.55, 0.47, 0.30, 0.22, 0.17, 0.14])
    t = np.array([5.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    d = 10
    Q = 50
    final = diffevol(Pop,cr,gam,pmut,0,ssr,h,models.slugmodel,[t,Q,d])
    #print final
    return final
testFunction()