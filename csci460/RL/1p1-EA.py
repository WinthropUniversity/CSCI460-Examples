import numpy as np
import sys
    
## Here is a simple (1+1)-EA on linear psuedo-Boolean functions
def ea(n, quiet=False):
    # Create a random linear function by just picking random weights for the whole run
    weights = np.random.uniform(size=n) 

    # Initialize
    x = np.array([0]*n)  # Start at all zeros
    generation = 0

    # Evaluate (just the some of the weights of the bits that are on)
    f = np.dot(x,weights)

    while np.sum(x) < n:  # Go until we flipped all bits
        generation += 1
        if not quiet:
            print(generation, x, f)

        # Mutate
        mask = np.random.choice([0,1], size=n, replace=True, p=[1-(1/n),1/n])
        xp = x ^ mask

        # Evaluate (just the some of the weights of the bits that are on)
        fp = np.dot(xp,weights)

        # Survival Selection
        if fp >= f:
            x = xp
            f = fp

    if not quiet:
        print("Solution: ", x, f)

    # Return the number of iterations it took before we saw the optimum for the first time
    return generation


def runIndependentTrials(nVals, numTrials):
    # Loop through every value of N
    for n in nVals:
        # Repeat that numTrials number of times
        for trial in range(numTrials):
            firstHittingTime = ea(n, True)
            print('{:1}, {:2}, {:3}'.format(n, trial, firstHittingTime))


if __name__ == "__main__":
    # Run with values for n from 2^3 to 2^12 by powers of 2
    nVals = np.power(2, range(3,12))
    print("n, trial, firstHit")
    runIndependentTrials(nVals, 30)


## Run this like this:
##   python3 1p1-EA.py > output.csv
##   Rscript plot1p1.r
##   Then go open the output.pdf file