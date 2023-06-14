# Import the game
import gym_super_mario_bros
# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# Import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# Import GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation
# Import Frame Stacker Wrapper Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# Import Matplotlib to show the impact of frame stacking
from matplotlib import pyplot as plt

# To avoid ValueError when unpakcing too many values apart from observations
class CustomDummyVecEnv(DummyVecEnv):
    def reset(self):
        for env_idx in range(self.num_envs):
            obs, *_ = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return self._obs_from_buf()

    def step_wait(self):

        for env_idx in range(self.num_envs):
            obs, rew, done, info = self.envs[env_idx].step(self.actions[env_idx])
            self.buf_rews[env_idx] = rew
            self.buf_dones[env_idx] = done
            self.buf_infos[env_idx] = info
            if done:
                obs, *_ = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), self.buf_rews, self.buf_dones, self.buf_infos)

# Wrapping the environment
# 1. Create the base environment
env = gym_super_mario_bros.make('SuperMarioBros-v1')
# 2. Simplify the controls 
env = JoypadSpace(env, SIMPLE_MOVEMENT)
"""# 3. Greyscale the environment
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = CustomDummyVecEnv([lambda: env])
# 5. Create the stacked frames
env = VecFrameStack(env, 4,channels_order='last')"""

# Import os for file path management
import os 
# Import our main Algorithm PPO 
from stable_baselines3 import PPO

# Load model
model = PPO.load('./train/first_model/best_model_100000')
# Start the game 
state = env.reset()
# Loop through the game
while True: 
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render(mode = 'rgb_array')
env.close()