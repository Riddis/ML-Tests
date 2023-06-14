# Import the game
import gym_super_mario_bros
# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# Import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

"""#SETTUP GAME
env = gym_super_mario_bros.make('SuperMarioBros-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Create a flag - restart or not
done = True
# Loop through each frame in the game
for step in range(500): 
    # Start the game to begin with 
    if done: 
        # Start the gamee
        env.reset()
    # Do random actions
    state, reward, done, info = env.step(env.action_space.sample())
    # Show the game on the screen
    env.render()
# Close the game
env.close()"""

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
# 3. Greyscale the environment
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = CustomDummyVecEnv([lambda: env])
# 5. Create the stacked frames
env = VecFrameStack(env, 4,channels_order='last')

state = env.reset()

# Import os for file path management
import os 
# Import our main Algorithm PPO 
from stable_baselines3 import PPO
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True
    
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

# Setup model saving callback
callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)

# This is the AI model started
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, 
            n_steps=512) 

# Train the AI model, this is where the AI model starts to learn
model.learn(total_timesteps=100000, callback=callback)

