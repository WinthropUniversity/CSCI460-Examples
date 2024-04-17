import numpy as np
import gym
import sys


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


def ChooseAction(Q, env, state, temp):
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
    gaugeIncr = maxNumSteps / 100

    # Reset the env
    state = env.reset()[0]

    # Take as many steps as requested
    for stepIdx in range(maxNumSteps):
        # A little progress gauge
        if stepIdx % gaugeIncr == 0:
            sys.stdout.write(".")
            sys.stdout.flush()

        # Choose an action, take a step
        action = ChooseAction(Q, env, state, temperature)
        newState, reward, done, truncated, info = env.step(action)

        # Use the observed reward and new state to update the Q table approx.
        UpdateQTable(Q, state, action, reward, discount, newState)

        # If the episode is over, reset the sim, otherwise update the state
        if done:
            state = env.reset()[0]
            temperature = max(0.99 * temperature, 0.01) # anneal temp
            #sys.stdout.write("G")
            #sys.stdout.flush()
        else:
            state = newState
            #sys.stdout.write(".")
            #sys.stdout.flush()
    print()


def main():
    # Ugly global variables
    #envName = "Taxi-v3"
    envName = "FrozenLake-v1"
    env = gym.make(envName, is_slippery=False, render_mode="ansi")
    env.reset()
    print(env.render())

    Q = ResetQTable(env)
    temperature = 100
    discount = 0.9
    maxNumSteps = 3000000

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
        print("  ", envState, "::", envAction, {0:"Left", 1:"Down", 2:"Right", 3:"Up", 4:"quit"}[envAction])
    print()

    env.reset()
    print(env.render())


if __name__ == "__main__":
    main()
