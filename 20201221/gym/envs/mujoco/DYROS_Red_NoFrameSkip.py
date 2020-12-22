from gym.envs.mujoco import DYROS_Red

class DYROSRedNoFrameSkipEnv(DYROS_Red.DYROSRedEnv):
    def __init__(self):
        DYROS_Red.DYROSRedEnv.__init__(self, frameskip=100)
