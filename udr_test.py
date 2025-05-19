"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse

import gym

from env.custom_hopper import *

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

from env.udr_wrapper import UDRWrapper



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--device', default='cuda', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=True, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=100, type=int, help='Number of test episodes')

    return parser.parse_args()

args = parse_args()


def main():

    source_env = gym.make('CustomHopper-source-v0')
    target_env = gym.make('CustomHopper-target-v0')
    udr_env = UDRWrapper(gym.make('CustomHopper-source-v0'))


    model = SAC.load('udr_model')

    '''
    print("\n--- udr → source ---")
    mean, std = evaluate_policy(model, source_env, n_eval_episodes=args.episodes, render=args.render)
    print(f"Return: {mean:.2f} ± {std:.2f}")
    '''
    print("\n--- udr → target ---")
    mean, std = evaluate_policy(model, target_env, n_eval_episodes=args.episodes, render=args.render)
    print(f"Return: {mean:.2f} ± {std:.2f}")
    '''
    print("\n--- udr → udr ---")
    mean, std = evaluate_policy(model, udr_env, n_eval_episodes=args.episodes, render=args.render)
    print(f"Return: {mean:.2f} ± {std:.2f}")
    '''
	

if __name__ == '__main__':
	main()