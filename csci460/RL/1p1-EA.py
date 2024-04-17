import numpy as np
import sys
    
def ea(n, quiet=False):
    # Initialize
    x = np.array([0]*n)  # Start at all zeros
    weights = np.random.uniform(size=n)
    generation = 0

    # Evaluate
    f = np.sum(x * weights)

    while np.sum(x) < n:  # Go until we flipped all bits
        generation += 1
        if not quiet:
            print(generation, x, f)

        # Mutate
        mask = np.random.choice([0,1], size=n, replace=True, p=[1-(1/n),1/n])
        xp = x ^ mask

        # Evaluate
        fp = np.sum(xp * weights)

        # Survival Selection
        if fp >= f:
            x = xp
            f = fp

    if not quiet:
        print("Solution: ", x, f)

    return generation


def runIndependentTrials(nVals, numTrials):
    for n in nVals:
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