import gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv

import numpy as np
import datetime
import torch as th

def main():
   # # multiprocess environment
   # n_cpu = 8
   # env = SubprocVecEnv([lambda: gym.make('DYROSRed-v1') for i in range(n_cpu)])
   # env = VecNormalize(env, norm_obs=True, clip_obs=2.0, norm_reward=False, training=True)

   # # n_cpu = 1   
   # # env = gym.make('DYROSRed-v1')
   # # env = DummyVecEnv([lambda: env])
   # # env = VecNormalize(env, norm_obs=True, clip_obs=2.0, norm_reward=False, training=True)

   # model = PPO('MlpPolicy', env, verbose=1, n_steps=int(4096/n_cpu), wandb_use=True)
   # model.learn(total_timesteps=60000000)
   # file_name = "ppo2_DYROSRed_" + str(datetime.datetime.now())
   # model.save(file_name)
   # env.save(file_name+"_env.pkl")

   # del model # remove to demonstrate saving and loading
   # del env

   file_name = "ppo2_DYROSRed_2020-12-25 03:58:53.805515"
   
   env = gym.make('DYROSRed-v1')
   env = DummyVecEnv([lambda: env])
   env = VecNormalize(env, norm_obs=True, clip_obs=2.0, norm_reward=False, training=False)
   env = env.load(file_name+"_env.pkl", env)

   model = PPO.load(file_name, env=env, wandb_use=False)

   #Enjoy trained agent
   obs =  np.copy(env.reset())
   epi_reward = 0

   while True:
      action, _states = model.predict(obs, deterministic=True)

      obs, rewards, dones, info = env.step(action)
      env.render()
      epi_reward += rewards      
      
      if dones:
         print("Episode Reward: ", epi_reward)
         epi_reward = 0
      env.render()
      
if __name__ == '__main__':
    main()