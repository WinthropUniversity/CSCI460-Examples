import numpy as np
import gym


def ResetQTable(env):
    """
    Create a (state-by-action) Q table, with initial values all set to zero
    """
    numStates = env.observation_space.n
    numActions = env.action_space.n
    return np.reshape( [0]*(numStates*numActions), (numStates, numActions) )


def UpdateQTable(Q, state, action, reward, discount, newState):
    """
    Update the (sate, action) based on the Q approximation update formula.
    """
    val = reward + discount * np.max(Q[newState])
    Q[state, action] = val


def ChooseAction(Q, state, temp):
    """
    Choose an action probabilistically using the Boltzman equation based
    on the values in the current approximation of the Q table.  The 'temp'
    parameter 
    """
    values = np.exp(Q[state] / temp)
    values = values / sum(values)

    actions = np.arange(start=0, stop=env.action_space.n)
    return np.random.choice(actions, size=None, replace=True, p=values)


def RunSimulation(env, maxNumSteps, Q, temperature, discount):
    # Reset the env
    state = env.reset()

    # Take as many steps as requested
    for _ in range(maxNumSteps):
        # Choose an action, take a step
        action = ChooseAction(Q, state, temperature)
        newState, reward, done, info = env.step(action)

        # Use the observed reward and new state to update the Q table approx.
        UpdateQTable(Q, state, action, reward, discount, newState)

        # If the episode is over, reset the sim, otherwise update the state
        if done:
            state = env.reset()
            #temperature = np.max(0.9 * temperature, 1) # anneal temp
        else:
            state = newState


# Ugly global variables
#envName = "Taxi-v3"
envName = "FrozenLake-v0"
env = gym.make(envName, is_slippery=False)

Q = ResetQTable(env)
temperature = 100
discount = 0.9
maxNumSteps = 1000000

# Learn, Agent, Learn!!
print()
print("Running our Q-Learner for", maxNumSteps, "steps on", envName, "...")
RunSimulation(env, maxNumSteps, Q, temperature, discount)

# Report the Q-Table now
print()
print("Q-Table:")
for s in range(Q.shape[0]):
    for a in range(Q.shape[1]):
        print("{:9.3f}".format(Q[s,a]), end='')
    print()

# Report the policy found:
print()
print("Best policy:")
for state in range(Q.shape[0]):
    envState  = state #env.observation_space[state]
    envAction = np.argmax(Q[state]) #env.action_space[np.argmax(Q[state])]
    print("  ", envState, "::", envAction)
print()