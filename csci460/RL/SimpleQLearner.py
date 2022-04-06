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


env = gym.make("Taxi-v3")
state = env.reset()
Q = ResetQTable(env)
temperature = 1
discount = 0.9

for _ in range(50000):
    action = ChooseAction(Q, state, temperature)
    newState, reward, done, info = env.step(action)
    UpdateQTable(Q, state, action, reward, discount, newState)

    if done:
        state = env.reset()
    else:
        state = newState

for s in range(500):
    for a in range(6):
        print(Q[s,a],'\t', end='')
    print()