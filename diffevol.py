##########################
# diffevol.py            #
# Differential Evolution #
#   by James V. Soukup   #
#   for CEE 290 HW #3    #
##########################

################################
# The models live in models.py #
################################
from models import *

#######################################
# Sum of Squared Residuals            #
#  (Cost Function)                    #
#  Parameters:                        #
#    f - Model Function               #
#    p - Model Function Parameters    #
#    d - Data to compare to the model #
#  Output:                            #
#    (d-f(p))'*(d-f(p))               #
#######################################
def ssr(f,p,d):
    return

###############################
# This is where I'd put other #
# cost functions              #
# ... if I had any...         #
###############################

#######################################
# diffevol function                   #
#  Parameters:                        #
#    f     - Cost Function            #
#    fpars - Cost Function Parameters #
#    pop   - Initial Population       #
#    cr    - Crossover Probability    #
#    pmut  - Mutation Probability     #
#######################################
def diffevol():
    return