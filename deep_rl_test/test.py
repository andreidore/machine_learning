import gym
import gym_miniworld
from gym_miniworld.wrappers import *
import cv2

from dqn import DQN


env = gym.make("MiniWorld-ThreeRooms-v0", obs_width=512, obs_height=512)
#env = gym.make("MiniWorld-Hallway-v0")

observation = env.reset()

print(env.action_space)

rl = DQN()


while True:

    env.render()

    # your agent goes here
    action = env.action_space.sample()

    print(action)

    observation, reward, done, info = env.step(0)

    # print(observation.shape)

    image = observation.copy()
    #image = cv2.resize(image, (512, 512))

    cv2.imshow('image', image)

    if done:
        break
