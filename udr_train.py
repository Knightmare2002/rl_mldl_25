"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *
from stable_baselines3.sac import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from env.udr_wrapper import UDRWrapper
import itertools


def tuning():

    learning_rates = [3e-4, 1e-4, 1e-3]
    gammas = [0.95, 0.98, 0.99]

    best_mean_reward = -float('inf')

    best_params = {}
    

    n_steps = 300_000

    udr_env = UDRWrapper(gym.make('CustomHopper-source-v0'))

    print('State space:', udr_env.observation_space)  # state-space
    print('Action space:', udr_env.action_space)  # action-space
    print('Dynamics parameters:', udr_env.get_parameters())  # masses of each link of the Hopper

    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #

    for lr, gamma in itertools.product(learning_rates, gammas):
        
        #-------------------------Source_env-----------------------------
        print(f'\ntraining on source_env: lr={lr}\tgamma={gamma}')

        source_model = SAC(MlpPolicy, udr_env, verbose=0, learning_rate=lr, gamma=gamma).learn(n_steps, progress_bar=True)

        mean_reward, _ = evaluate_policy(source_model, udr_env, n_eval_episodes=10, warn=False)

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_params = {'learning_rate': lr, 'gamma':gamma}
            source_model.save("source_udr_model")


    print("\n Grid Search Completed")
    print(f'Best hyperparameters source_model: {best_params}')

    print(f"Best Source Reward: {best_mean_reward:.2f}")

    return best_params

def train():
    param = tuning()

    n_steps = 3_000_000

    lr = param['learning_rate']
    gamma = param['gamma']

    udr_env = UDRWrapper(gym.make('CustomHopper-source-v0'))

    final_model = SAC(MlpPolicy, udr_env, verbose=0, learning_rate=lr, gamma=gamma).learn(n_steps, progress_bar=True)
    final_model.save("udr_model")



if __name__ == '__main__':
    train()  