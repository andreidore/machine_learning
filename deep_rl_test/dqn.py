import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from drl import DRL


class DQN(DRL):

    def __init__(self):
        super().__init__()

        self.model = self.build_model()

        self.gamma = 0.95

        self.epsilon = 1.0

        self.epsilon_decay = 0.995

        self.epsilon_min = 0.1

        self.memory_buffer = deque(maxlen=2000)

    def build_model(self):

        inputs = Input(4)
        x = Dense(16, activation="relu")(inputs)
        x = Dense(16, activation="relu")(x)
        x = Dense(2, activation="linear")(x)

        model = Model(inputs=inputs, outputs=x)

        model.compile(loss="mse", optimizer=Adam(1e-3))

        return model

    def update_epsilon(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def egreedy_action(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randint(0, 1)

        q_values = self.model.predict(state)[0]

        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):

        self.memory_buffer.append((state, action, reward, next_state, done))

    def process_batch(self, batch):

        data = random.sample(self.memory_buffer, batch)

        # print(len(data))

        states = np.array([d[0] for d in data])
        next_states = np.array([d[3] for d in data])

        #print(states.shape, next_states.shape)

        y = self.model.predict(states)
        q = self.model.predict(next_states)

        # print(y.shape)
        # exit()

        for i, (_, action, reward, _, done) in enumerate(data):
            target = reward
            if not done:
                target += self.gamma*np.amax(q[i])

            y[i][action] = target

        return states, y

    def train(self, episodes, batch):

        history = {"episode": [], "episode_reward": [], "loss": []}

        count = 0
        for i in range(episodes):

            observations = self.env.reset()

            reward_sum = 0
            loss = np.infty
            done = False

            while not done:

                x = observations.reshape(-1, 4)

                action = self.egreedy_action(x)

                observations, reward, done, _ = self.env.step(action)

                reward_sum += reward

                self.remember(x[0], action, reward, observations, done)

                if len(self.memory_buffer) > batch:
                    X, y = self.process_batch(batch)

                    loss = self.model.train_on_batch(X, y)

                    count += 1

                    self.update_epsilon()

            if i % 5 == 0:
                history["episode"].append(i)
                history["episode_reward"].append(reward_sum)
                history["loss"].append(loss)

                print('Episode: {} | Episode reward: {} | loss: {:.3f} | e:{:.2f}'.format(
                    i, reward_sum, loss, self.epsilon))

        self.model.save_weights('model/dqn.h5')

        return history


if __name__ == "__main__":

    dqn = DQN()

    print("Train")
    history = dqn.train(70, 32)
    dqn.save_history(history, 'dqn.csv')

    dqn.load()
    dqn.play()
