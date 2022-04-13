import numpy as np
import gym


envName = "FrozenLake-v0"
# https://gym.openai.com/envs/FrozenLake-v0/
env = gym.make(envName, is_slippery=False)
state = env.reset()

while (True):
    env.render()

    print()
    print("You are in state: ", state)
    print("Actions include [0-{:d}]".format(env.action_space.n))
    print("  0-Left, 1-Down, 2-Right, 3-Up, 4-quit")
    action = int(input("Choose Action: "))

    if (action == 4):
        break  # Very, very ugly code here

    state, reward, done, info = env.step(action)

    if reward > 0:
        env.render()
        print("GOOOAAALLLL!!!!  You got rewarded {:f}!".format(reward))
        state = env.reset()

    elif done:
        env.render()
        print("Oops.  You fell in a hole.  Start again!")
        state = env.reset()
