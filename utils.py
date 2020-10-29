import numpy as np

#env = gym.make('FrozenLake-v0')


def choose_action(observation, Q, env, epsilon):
  action = 0
  if np.random.uniform(0, 1) < epsilon:
    action = env.action_space.sample()
  else:
    action = np.argmax(Q[observation, :])
  return action

def learn(observation, observation2, reward, action, Q, learning_rate, gamma):
  prediction = Q[observation, action]
  target = reward + gamma * np.max(Q[observation2, :])
  Q[observation, action] = Q[observation, action] + learning_rate * (target - prediction)