import numpy as np
import gym

class UDRWrapper(gym.Wrapper):
    """
    Uniform Domain Randomization Wrapper for Hopper
    Randomizes the masses of all body parts except the torso at every reset.
    """

    def __init__(self, env, mass_range=(0.5, 1.5), seed=42):
        super().__init__(env)
        self.mass_range = mass_range
        self.rng = np.random.RandomState(seed)

        self.default_masses = np.copy(self.env.sim.model.body_mass)
        self.randomize_ids = [
            i for i, name in enumerate(self.env.sim.model.body_names)
            if name not in [b'torso', b'world']
        ]

    def reset(self, **kwargs):
        new_masses = np.copy(self.default_masses)
        for i in self.randomize_ids:
            scale = self.rng.uniform(*self.mass_range)
            new_masses[i] *= scale

        self.env.sim.model.body_mass[:] = new_masses

        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        return self.env.step(action)
