import gym
import numpy as np
import copy
import random
import math
import queue
import time
from gym import spaces, error
import airsim
import cv2
import os
import setup_path
import time
import matplotlib.pyplot as plt
from PIL import Image




class CarEnv(gym.Env):
    def __init__(self):
        self.client = airsim.CarClient("10.8.105.156")
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.car_controls = airsim.CarControls()
        self.action_space = spaces.Discrete(7)
        self.time_step = 0 
        self.x_pos_goal = 0.6   #0.545647
        self.y_pos_goal = -2.5  #-1.419126
        self.z_pos_goal = 0.2   #0.176768
        self.counter_no_state = 0 
        self.w_rot_goal = 1.0    # 0.999967
        self.x_rot_goal = 0.0    # 
        self.y_rot_goal = 0.0    # -0.000095
        self.z_rot_goal = 0.02    # 0.019440
        self.max_step = 10  # steps to check if blocked 
        self.last_states = queue.deque(maxlen=self.max_step)        
        self.height = 84
        self.width = 84  # old 320 
        # self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.observation_space = spaces.Box(low = 0, high = 255, shape=(self.height, self.width, 2))
        self.debug_mode = False
        self.goal_counter = 0
        self.count = 0

    def reset(self):
        """
        This function resets the environment and returns the game state.
        """
        self.client.reset()
        
        #state2 = self._get_state("4") # backward camera
        #state = np.stack((state1, state2), axis=2)
        for i in range(11):
            state1, reward, done,_ = self.step(random.randint(0, 6))
        state1, reward, done,_ = self.step(3)
        state = self._get_state("3")  # forward camera
        pose = self.client.simGetVehiclePose()
        reward = self._get_reward(pose)
        self.time_step = 0 
        return np.array(state)

        

    def render(self, responses, mode='human', close=False):
        """
        This function renders the current game state in the given mode.
        """
        wait = 0.2
        for response in responses:
            img1d = np.fromstring(response.image_data_uint8, dtype= np.uint8) # get numpy array
            if img1d.shape[0] == 268800:
                img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 3 channel image array H X W>
                #cv2.imshow("video", img_rgb)
                im = plt.imshow(img_rgb)
                im.set_data(img_rgb)
                plt.pause(wait)



    def step(self, action):
        """

        Parameters
        ----------
        action : int
            The action is an angle between 0 and 180 degrees, that
            decides the direction of the bubble.

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing the
                state of the environment.
            reward (float) :
                amount of reward achieved by the previous action.
            episode_over (bool) :
                whether it's time to reset the environment again.
            info (dict) :
                diagnostic information useful for debugging.
        """
        self.time_step += 1 
        self.car_controls.brake = 0
        if action == 0:
            # go forward
            self.car_controls.throttle = 0.5
            self.car_controls.steering = 0
            self.client.setCarControls(self.car_controls)
        
        if action == 1:
            # Go forward + steer right
            self.car_controls.throttle = 0.5
            self.car_controls.steering = 1
            self.client.setCarControls(self.car_controls)
    
        if action == 2:
            # Go forward + steer left
            self.car_controls.throttle = 0.5
            self.car_controls.steering = -1
            self.client.setCarControls(self.car_controls)
    
        if action == 3:
            # Go stop
            self.car_controls.throttle = 0
            self.car_controls.steering = 0
            self.car_controls.brake = 1
            self.client.setCarControls(self.car_controls)
    
        if action == 4:
            # Go backward
            self.car_controls.throttle = -0.5
            self.car_controls.steering = 0
            self.client.setCarControls(self.car_controls)
        
        if action == 5:
            # Go backward + steer right
            self.car_controls.throttle = -0.5
            self.car_controls.steering = 1
            self.client.setCarControls(self.car_controls)
        
        if action == 6:
            # Go backward + steer left
            self.car_controls.throttle = -0.5
            self.car_controls.steering = -1
            self.client.setCarControls(self.car_controls)

        pose = self.client.simGetVehiclePose()
        reward = self._get_reward(pose)
        done  = False
        state1 = self._get_state("3")  # forward camera
        #state2 = self._get_state("4") # backward camera

        #state = np.stack((state1, state2), axis=2)
        state = state1
        #if reward > -2:
        #    reward = reward + 2 
        reward /= 10
        if self._is_goal(pose):
            reward = 1
        else:
            reward  = max(reward, -3) 
        return state, reward, done, pose 
    
    def process_image(self, state):
        """ 
        """
        debug_mode = False
        if debug_mode:
            img = Image.fromarray(state, 'RGB')
            img.show()
            img.close()

        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = cv2.resize(state,(self.width, self.height))
        # normailze (values between 0 and 1)
        # state = state / 255
        return state

    
    def _get_reward(self, pose):
        """
        This function calculates the reward.
        """
        x_pos = pose.position.x_val
        y_pos = pose.position.y_val
        z_pos = pose.position.z_val
        
        x_rot = pose.orientation.x_val
        y_rot = pose.orientation.y_val
        z_rot = pose.orientation.z_val

        # calculate difference between current and goal
        dif = 0
        x_dif = math.sqrt((x_pos - self.x_pos_goal)**2)
        y_dif = math.sqrt((y_pos - self.y_pos_goal)**2)
        z_dif = math.sqrt((z_pos - self.z_pos_goal)**2)

        x_r_dif = math.sqrt((x_rot - self.x_rot_goal)**2)
        y_r_dif = math.sqrt((y_rot - self.y_rot_goal)**2)
        z_r_dif = math.sqrt((z_rot - self.z_rot_goal)**2)
        dif = x_dif + y_dif + z_dif + x_r_dif + y_r_dif + z_r_dif
        return -dif



    def _is_goal(self, pose):
        x_pos = pose.position.x_val
        y_pos = pose.position.y_val
        z_pos = pose.position.z_val
        
        x_rot = pose.orientation.x_val
        y_rot = pose.orientation.y_val
        z_rot = pose.orientation.z_val

        # calculate difference between current and goal
        
        x_dif = math.sqrt((x_pos - self.x_pos_goal)**2)
        y_dif = math.sqrt((y_pos - self.y_pos_goal)**2)
        z_dif = math.sqrt((z_pos - self.z_pos_goal)**2)

        x_r_dif = math.sqrt((x_rot - self.x_rot_goal)**2)
        y_r_dif = math.sqrt((y_rot - self.y_rot_goal)**2)
        z_r_dif = math.sqrt((z_rot - self.z_rot_goal)**2)
        eps = 0.25
        if self.debug_mode:
            debug_message = '                                position difference of x: {:.2f}, '
            debug_message += 'y: {:.2f}, z: {:.2f}, x_r: '
            debug_message += '{:.2f}, y_r: {:.2f}, z_r: {:.2f}' 
            debug_message = debug_message.format(x_dif, y_dif, z_dif, x_r_dif, y_r_dif , z_r_dif)
            print(debug_message, end='\r', flush=True) 
        if x_dif < eps and y_dif < eps and z_dif < eps and x_r_dif < eps  and y_r_dif < eps  and z_r_dif < eps:
            self.goal_counter +=1
            return True
        return False 

        

    def _is_over(self):
        """
        Returns a string and a bool in which way the game is
        over or not.
        """
        if len(self.last_states) < self.max_step:
            return False
        # check if positon changed
        allowed_error = 0.1
        the_same = 0
        for i in range(1,self.max_step):
            if abs(self.last_states[0].position.x_val- self.last_states[i].position.x_val) <= allowed_error:
                the_same += 1
            if abs(self.last_states[0].position.y_val- self.last_states[i].position.y_val) <= allowed_error:
                the_same += 1
            if abs(self.last_states[0].position.z_val- self.last_states[i].position.z_val) <= allowed_error:
                the_same += 1
        if the_same >=25:
            return True
        return False

    def _get_state(self, camera):
        """ if camera is 3 forward and 4 backward
        """
        # get back camera 
        responses = self.client.simGetImages([airsim.ImageRequest(camera,
            airsim.ImageType.DepthVis),
            airsim.ImageRequest("3", airsim.ImageType.DepthPerspective, True), #depth in pe>
            airsim.ImageRequest(camera, airsim.ImageType.Scene), #scene
            airsim.ImageRequest(camera, airsim.ImageType.Scene, False, False)])
        state_exists = False 
        for response in responses:
            img1d = np.fromstring(response.image_data_uint8, dtype= np.uint8) # get numpy array
            if img1d.shape[0] == 268800:
                state_exists = True
                state= img1d.reshape(response.height, response.width, 3) # reshape array to 3 channel image
                #img = Image.fromarray(state, 'RGB')
                #text = 'my-{}.png'.format(self.count)
                #self.count +=1
                #img.save(text)
                # state = self.process_image(state) 
        if state_exists == False:
            state = np.zeros((140, 640, 3), dtype= np.uint8)
        return state
        

    def _is_over(self):
        """
        Returns a string and a bool in which way the game is
        over or not.
        """
        if len(self.last_states) < self.max_step:
            return False
        # check if positon changed
        allowed_error = 0.1
        the_same = 0
        for i in range(1,self.max_step):
            if abs(self.last_states[0].position.x_val- self.last_states[i].position.x_val) <= allowed_error:
                the_same += 1
            if abs(self.last_states[0].position.y_val- self.last_states[i].position.y_val) <= allowed_error:
                the_same += 1
            if abs(self.last_states[0].position.z_val- self.last_states[i].position.z_val) <= allowed_error:
                the_same += 1
        if the_same >=25:
            return True
        return False
