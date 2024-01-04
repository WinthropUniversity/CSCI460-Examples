import numpy as np
import gym


envName = "FrozenLake-v1"
# https://gym.openai.com/envs/FrozenLake-v0/
env = gym.make(envName, is_slippery=False, render_mode="ansi")
state = env.reset()[0]

while (True):
    print(env.render())

    print()
    print("You are in state: ", state)
    print("Actions include [0-{:d}]".format(env.action_space.n))
    print("  0-Left, 1-Down, 2-Right, 3-Up, 4-quit")
    action = int(input("Choose Action: "))

    if (action == 4):
        break  # Very, very ugly code here

    state, reward, done, truncate, info = env.step(action)

    if reward > 0:
        env.render()
        print("GOOOAAALLLL!!!!  You got rewarded {:f}!".format(reward))
        state = env.reset()[0]

    elif done:
        print(env.render())
        print("Oops.  You fell in a hole.  Start again!")
        state = env.reset()
