from gym import spaces
import gym
import numpy as np 

class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)
        self.action_space = spaces.MultiDiscrete([3,3,2,3,3,2]) #upp,ner,vänster,höger,hopp,kamera
        self.org =env
    def action(self, act):
        action = self.org.action_space.noop()
        if act[0] == 1:
            action['forward'] = 1
        if act[0] == 2:
            action['back'] = 1
        if act[1] == 1:
            action['left'] = 1
        if act[1] == 2:
            action['right'] = 1
        action['jump'] = act[2]
        action['camera'] = [(act[4]-1)*5., (act[3]-1)*10.]
        action['attack'] = act[5]
        return action

class ObsWrapper(gym.ObservationWrapper):
    def __init__(self, shape, env):
        super(ObsWrapper, self).__init__(env)
        self.org = env
        self.shape = shape
        channels = 2
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.shape[0], self.shape[1], channels), dtype=np.float)            

    def observation(self, observation):
        import cv2
        pov_obseration = observation['pov']
        compass_angle = observation['compassAngle']
        
        obs = cv2.cvtColor(pov_obseration, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, self.shape, interpolation = cv2.INTER_AREA)
        obs = obs / 255
        compass_angle_scale = 180
        compass_scaled = compass_angle / compass_angle_scale
        compass_channel = np.ones(shape=self.shape, dtype=obs.dtype) * compass_scaled
        
        obs = np.concatenate([obs[...,None], compass_channel[...,None]], axis=-1)
        #obs = np.moveaxis(obs, [0, 1, 2], [1, 2, 0])
        return obs