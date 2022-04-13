from sys import maxunicode
import numpy as np
import gym


def InitializeIndividual(env):
    numStates    = env.observation_space.n
    numActions   = env.action_space.n
    bitsPerBlock = int(np.ceil(np.log2(numActions)))

    # Draw a random binary array large enough to represent
    # an action choice for every state
    x = np.random.choice([0,1], replace=True, size=bitsPerBlock*numStates)

    return (x)


def MutateIndividual(parent, pm):
    # Copy the parent
    child = parent.copy()

    # Flip each bit in the child with probability pm, i.i.d.
    for idx in range(len(child)):
        if (np.random.uniform() < pm):
            child[idx] = np.abs(1 - child[idx])

    return child


def DecodeIndividual(env, individual):
    numStates    = env.observation_space.n
    numActions   = env.action_space.n
    bitsPerBlock = int(np.ceil(np.log2(numActions)))

    # Initialize a rule dictionaryt to lookup actions given states
    ruleDict = {}

    # Let's figure out what each action should be
    for state in range(numStates):
        # Convert the block of bits to an integer
        action = 0
        for idx in range(state*bitsPerBlock, (state+1)*bitsPerBlock):
            action = action << 1
            action += individual[idx]

        # Now make sure that integer remains in the action space
        ruleDict[state] = action % numActions

    return ruleDict


def EvaluateIndividual(env, individual, maxNumSteps):
    fitness = 0
    agentPolicy = DecodeIndividual(env, individual)

    # Run the sim
    state = env.reset()
    for _ in range(maxNumSteps):
        action = agentPolicy[state]
        newState, reward, done, info = env.step(action)
        fitness += reward

        # If the episode is over, reset the sim, otherwise update the state
        if done:
            state = env.reset()
        else:
            state = newState

    # Return the average fitness per step
    return float(fitness) / float(maxNumSteps)


def GetIndividualAsString(individual):
    bitString = ''
    for bit in individual:
        if bit == 0:
            bitString += '0'
        else:
            bitString += '1'
    return bitString


# Ugly global variables ...
maxNumSteps = 1000
maxNumGenerations = int(1000000/maxNumSteps) # To be fair, use the same total number as in QL

#envName = "Taxi-v3"
envName = "FrozenLake-v0"
env = gym.make(envName, is_slippery=False)

# Initialize the first individual
parent = InitializeIndividual(env)
parentFitness = EvaluateIndividual(env, parent, maxNumSteps)

# Make the mutation rate 1/n
n = len(parent)
pm = 1.0/float(n)

# Report some useful info
print()
print("Running an EA reinforcement learer on", envName, ",  n =", len(parent), ",  pm=", pm)
print("Max num steps per eval =", maxNumSteps, ",  Max Generations=", maxNumGenerations)
print()

for genCount in range(maxNumGenerations):
    print("Gen:", genCount, ",  Fitness:", parentFitness, ",  ", GetIndividualAsString(parent))

    child = MutateIndividual(parent, pm)
    childFitness = EvaluateIndividual(env, child, maxNumSteps)

    if (childFitness >= parentFitness):
        parent = child
        parentFitness = childFitness

# Report the best policy
print()
print("Best policy:")
bestPolicy = DecodeIndividual(env, parent)
for state in bestPolicy:
    envState  = state #env.observation_space[state]
    envAction = bestPolicy[state] #env.action_space[bestPolicy[state]]
    print("  ", envState, "::", envAction)
print("Best Fitness:", parentFitness)
print()