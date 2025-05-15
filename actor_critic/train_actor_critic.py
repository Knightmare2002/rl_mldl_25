"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse

import torch
import gym
from tqdm import tqdm

from env.custom_hopper import *
from agent_actor_critic import Agent, Policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=20000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()


def main():

    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())


    """
        Training
    """
    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, device=device)

    num_episodes = 1000
    print_every = 100
    #
    # TASK 2 and 3: interleave data collection to policy updates
    #

    for episode in tqdm(range(num_episodes)):
        done = False
        train_reward = 0
        state = env.reset()  # Reset the environment and observe the initial state

        while not done:  # Loop until the episode is over

            action, action_probabilities = agent.get_action(state)
            previous_state = state

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            #---Task 3---
           
            agent.update_policy(previous_state, state, action_probabilities, reward, done)  
            

            train_reward += reward

        

        if (episode+1)%print_every == 0:
            print('Training episode:', episode)
            print('Episode return:', train_reward)


    torch.save(agent.policy.state_dict(), "reinforce_model.mdl")

	

if __name__ == '__main__':
	main()