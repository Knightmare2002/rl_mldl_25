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
import itertools

SEED = 42


def tuning():

    learning_rates = [3e-4, 1e-4, 1e-3]
    gammas = [0.95, 0.98, 0.99]

    best_mean_reward_source = -float('inf')
    best_mean_reward_target = -float('inf')

    best_params_source = {}
    best_params_target = {}
    

    n_steps = 1000

    source_env = gym.make('CustomHopper-source-v0')
    target_env = gym.make('CustomHopper-target-v0')

    print('State space:', source_env.observation_space)  # state-space
    print('Action space:', source_env.action_space)  # action-space
    print('Dynamics parameters:', source_env.get_parameters())  # masses of each link of the Hopper

    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #

    for lr, gamma in itertools.product(learning_rates, gammas):
        
        #-------------------------Source_env-----------------------------
        print(f'\ntraining on source_env: lr={lr}\tgamma={gamma}')

        source_model = SAC(MlpPolicy, source_env, verbose=0, learning_rate=lr, gamma=gamma, seed=SEED).learn(n_steps, progress_bar=True)

        mean_reward, _ = evaluate_policy(source_model, source_env, n_eval_episodes=10, warn=False)

        if mean_reward > best_mean_reward_source:
            best_mean_reward_source = mean_reward
            best_params_source = {'learning_rate': lr, 'gamma':gamma}
            source_model.save("sac_source_model")

        #-------------------------Target_env-----------------------------
        print(f'training on target_env: lr={lr}\tgamma={gamma}')
        target_model = SAC(MlpPolicy, target_env, verbose=0, learning_rate=lr, gamma=gamma, seed=SEED).learn(n_steps, progress_bar=True)

        mean_reward, _ = evaluate_policy(target_model, target_env, n_eval_episodes=10, warn=False)

        if mean_reward > best_mean_reward_target:
            best_mean_reward_target = mean_reward
            best_params_target = {'learning_rate': lr, 'gamma':gamma}
            target_model.save("sac_target_model")

    print("\n Grid Search Completed")
    print(f'Best hyperparameters source_model: {best_params_source}')
    print(f'Best hyperparameters target_model: {best_params_target}')

    print(f"Best Source Reward: {best_mean_reward_source:.2f}")
    print(f"Best Target Reward: {best_mean_reward_target:.2f}")

    return best_params_source, best_params_target

def train():
    param_source, param_target = tuning()

    lr_source = param_source['learning_rate']
    lr_target = param_target['learning_rate']

    gamma_source = param_source['gamma']
    gamma_target = param_target['gamma']

    n_steps = 100_000

    source_env = gym.make('CustomHopper-source-v0')
    target_env = gym.make('CustomHopper-target-v0')

    source_model = SAC(MlpPolicy, source_env, verbose=0, learning_rate=lr_source, gamma=gamma_source, seed=SEED).learn(n_steps, progress_bar=True)
    source_model.save("sac_source_model")


    target_model = SAC(MlpPolicy, target_env, verbose=0, learning_rate=lr_target, gamma=gamma_target, seed=SEED).learn(n_steps, progress_bar=True)
    target_model.save("sac_target_model")


if __name__ == '__main__':
    train()  