import gym_runner
env = gym_runner.make('Skiing-ram-v0')
observation = env.reset()
episode = 1
reward_on_episode = 0
for i in range(1000):
    if i % 100 == 0:
        print("Iteration %d; running reward  = %.2f" % (i, reward_on_episode))
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    reward_on_episode += reward
    if done:
        print("DONE episode %d; reward obtained = %.2f" % (episode, reward_on_episode))
        episode += 1
        observation = env.reset()
