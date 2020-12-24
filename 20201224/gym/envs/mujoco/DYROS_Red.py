import numpy as np
from math import atan2
from gym.envs.mujoco import mujoco_env
from gym import utils
from gym.utils.cubic import cubic, cubicDot
import json
from math import exp
from pyquaternion import Quaternion
import mujoco_py

CollisionCheckBodyList = ["base_link",\
            "R_HipRoll_Link", "R_HipCenter_Link", "R_Thigh_Link", "R_Knee_Link",\
            "L_HipRoll_Link", "L_HipCenter_Link", "L_Thigh_Link", "L_Knee_Link",\
            "Waist1_Link", "Waist2_Link", "Upperbody_Link", \
            "R_Shoulder1_Link", "R_Shoulder2_Link", "R_Shoulder3_Link", "R_Armlink_Link", "R_Elbow_Link", "R_Forearm_Link", "R_Wrist1_Link", "R_Wrist2_Link",\
            "L_Shoulder1_Link", "L_Shoulder2_Link", "L_Shoulder3_Link", "L_Armlink_Link", "L_Elbow_Link", "L_Forearm_Link", "L_Wrist1_Link","L_Wrist2_Link"]

ObsBodyList = ["R_Thigh_Link", "R_Knee_Link","R_AnkleCenter_Link", \
            "L_Thigh_Link", "L_Knee_Link", "L_AnkleCenter_Link", \
            "Waist1_Link", "Upperbody_Link", \
            "R_Shoulder1_Link", "R_Armlink_Link", "R_Forearm_Link", "R_Wrist1_Link", "R_Foot_Link",\
            "L_Shoulder1_Link", "L_Armlink_Link", "L_Forearm_Link", "L_Wrist1_Link", "L_Foot_Link"]

Kp = np.asarray([400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, \
                400, 400, 400,\
                400, 400, 400, 400, 400, 400, 400, 400])

Kd = np.asarray([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, \
                10, 10, 10,\
                10, 10, 10, 10, 10, 10, 10, 10])

class DYROSRedEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, frameskip=66):
        mujoco_env.MujocoEnv.__init__(self, 'dyros_red.xml', frameskip)
        utils.EzPickle.__init__(self)
        for id in CollisionCheckBodyList:
            self.collision_check_id.append(self.model.body_name2id(id))
        print("Collision Check ID", self.collision_check_id)

    def _get_obs(self):
        mocap_cycle_dt = 0.033332
        mocap_data_num = 38
        mocap_cycle_period = mocap_data_num* mocap_cycle_dt
        phase = np.array((self.init_mocap_data_idx + self.time % mocap_cycle_period / mocap_cycle_dt) % mocap_data_num / mocap_data_num)

        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        basequat = Quaternion(self.sim.data.get_body_xquat("base_link"))
        basepos = self.get_body_com("base_link")
        basevel = self.sim.data.get_body_xvelp("base_link")
        baseangvel = self.sim.data.get_body_xvelr("base_link")

        return np.concatenate([phase.flatten(),
                            qpos[3:].flatten(),
                            qvel.flatten(),
                            basepos[2].flatten(),
                            [self.r_contact], [self.l_contact]])

    def step(self, a):
        mocap_cycle_dt = 0.033332
        mocap_data_num = 38
        mocap_cycle_period = mocap_data_num* mocap_cycle_dt

        self.time += self.dt

        local_time = self.time % mocap_cycle_period
        local_time_plus_init = (local_time + self.init_mocap_data_idx*mocap_cycle_dt) % mocap_cycle_period
        cycle_iter = int((self.init_mocap_data_idx + int(self.time / mocap_cycle_dt)) / mocap_data_num)
        self.mocap_data_idx = (self.init_mocap_data_idx + int(local_time / mocap_cycle_dt)) % mocap_data_num
        next_idx = self.mocap_data_idx + 1 

        if (cycle_iter != 0) and (self.mocap_data_idx == self.init_mocap_data_idx):
            self.cycle_init_root_pos[0] = self.sim.data.qpos[0]
            self.cycle_init_root_pos[1] = self.sim.data.qpos[1]
        
        target_data_qpos = np.zeros_like(a)
        target_data_qvel = np.zeros_like(a)
        target_data_body_delta = np.zeros(3)
        target_data_body_vel = np.zeros(3)

        for i in range(a.size):
            target_data_qpos[i] = cubic(local_time_plus_init, self.mocap_data[self.mocap_data_idx,0], self.mocap_data[next_idx,0], self.mocap_data[self.mocap_data_idx,i+8], self.mocap_data[next_idx,i+8], 0.0, 0.0)
            target_data_qvel[i] =  (self.mocap_data[next_idx,i+8] -  self.mocap_data[self.mocap_data_idx,i+8]) / mocap_cycle_dt

        
        if(self.mocap_data_idx >= self.init_mocap_data_idx):
            target_data_body_delta[0] = cubic(local_time_plus_init, self.mocap_data[self.mocap_data_idx,0], self.mocap_data[next_idx,0], self.mocap_data[self.mocap_data_idx,1] - self.mocap_data[self.init_mocap_data_idx,1], self.mocap_data[next_idx,1]-self.mocap_data[self.init_mocap_data_idx,1], 0.0, 0.0)
            target_data_body_delta[1] = cubic(local_time_plus_init, self.mocap_data[self.mocap_data_idx,0], self.mocap_data[next_idx,0], self.mocap_data[self.mocap_data_idx,3] - self.mocap_data[self.init_mocap_data_idx,3], self.mocap_data[next_idx,3]-self.mocap_data[self.init_mocap_data_idx,3], 0.0, 0.0)
            target_data_body_delta[2] = cubic(local_time_plus_init, self.mocap_data[self.mocap_data_idx,0], self.mocap_data[next_idx,0], self.mocap_data[self.mocap_data_idx,2] - self.mocap_data[self.init_mocap_data_idx,2], self.mocap_data[next_idx,2]-self.mocap_data[self.init_mocap_data_idx,2], 0.0, 0.0)
        else:
            target_data_body_delta[0] = cubic(local_time, self.mocap_data[37,0] + self.mocap_data[self.mocap_data_idx,0], self.mocap_data[37,0] + self.mocap_data[next_idx,0], self.mocap_data[37,1] + self.mocap_data[self.mocap_data_idx,1] - self.mocap_data[self.init_mocap_data_idx,1], self.mocap_data[37,1] + self.mocap_data[next_idx,1] - self.mocap_data[self.init_mocap_data_idx,1], 0.0, 0.0)
            target_data_body_delta[1] = cubic(local_time, self.mocap_data[37,0] + self.mocap_data[self.mocap_data_idx,0], self.mocap_data[37,0] + self.mocap_data[next_idx,0], self.mocap_data[self.mocap_data_idx,3] - self.mocap_data[self.init_mocap_data_idx,3], self.mocap_data[next_idx,3] - self.mocap_data[self.init_mocap_data_idx,3], 0.0, 0.0)
            target_data_body_delta[2] = cubic(local_time, self.mocap_data[37,0] + self.mocap_data[self.mocap_data_idx,0], self.mocap_data[37,0] + self.mocap_data[next_idx,0], self.mocap_data[self.mocap_data_idx,2] - self.mocap_data[self.init_mocap_data_idx,2], self.mocap_data[next_idx,2] - self.mocap_data[self.init_mocap_data_idx,2], 0.0, 0.0)
        
        target_data_body_vel[0] = (self.mocap_data[next_idx,1] - self.mocap_data[self.mocap_data_idx,1])/mocap_cycle_dt
        target_data_body_vel[1] = (self.mocap_data[next_idx,3] - self.mocap_data[self.mocap_data_idx,3])/mocap_cycle_dt
        target_data_body_vel[2] = (self.mocap_data[next_idx,2] - self.mocap_data[self.mocap_data_idx,2])/mocap_cycle_dt

        for i in range(self.frame_skip):
            qpos = self.sim.data.qpos
            qvel = self.sim.data.qvel
            torque = 400*(target_data_qpos + a - qpos[7:]) + 40*(- qvel[6:])
            self.do_simulation(torque,1)

        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        basequat = self.sim.data.get_body_xquat("base_link")
        basequat_desired = np.array([1,0,0,0]) #Quaternion(self.mocap_data[self.mocap_data_idx,4:8])
        baseQuatError = (1-np.dot(basequat_desired,basequat))

        Tar_Body = self.cycle_init_root_pos+target_data_body_delta

        # self.set_state(
        #     np.concatenate((Tar_Body, basequat_desired.elements, target_data_qpos)),
        #     self.init_qvel + np.concatenate((target_data_body_vel, np.zeros(3), target_data_qvel)),
        # )        
        # self.sim.step()

        # for i in range(self.frame_skip):
        #     qpos = self.sim.data.qpos
        #     qvel = self.sim.data.qvel
        #     torque = 400*(target_data_qpos - qpos[7:]) + 40*(target_data_qvel- qvel[6:])
        #     self.do_simulation(torque,1)
        
        done_by_contact = False
        self.r_contact = False
        self.l_contact = False
        if self.done_init is False:
            done_by_contact = False
            self.done_init = True
        else:
            for i in range(self.sim.data.ncon):
                if (self.sim.data.contact[i].geom1 == 0 and  any(self.model.geom_bodyid[self.sim.data.contact[i].geom2] == collisioncheckid for collisioncheckid in self.collision_check_id)) or \
                    (self.sim.data.contact[i].geom2 == 0 and any(self.model.geom_bodyid[self.sim.data.contact[i].geom1] == collisioncheckid for collisioncheckid in self.collision_check_id)):
                    done_by_contact = True
                if (self.sim.data.contact[i].geom1 == 0 and  self.model.geom_bodyid[self.sim.data.contact[i].geom2] == 8) or \
                    (self.sim.data.contact[i].geom2 == 0 and self.model.geom_bodyid[self.sim.data.contact[i].geom1] == 8):
                    self.r_contact = True
                if (self.sim.data.contact[i].geom1 == 0 and  self.model.geom_bodyid[self.sim.data.contact[i].geom2] == 15) or \
                    (self.sim.data.contact[i].geom2 == 0 and self.model.geom_bodyid[self.sim.data.contact[i].geom1] == 15):
                    self.l_contact = True

        if (self.mocap_data_idx == 37 or self.mocap_data_idx == 0 or self.mocap_data_idx == 1 or self.mocap_data_idx == 18 or self.mocap_data_idx == 19 or self.mocap_data_idx == 20):
            if (self.r_contact is True and self.l_contact is True):
                mimic_contact_reward = 0.2
            else:
                mimic_contact_reward = 0.0
        elif (self.mocap_data_idx <= 18):
            if (self.r_contact is True and self.l_contact is False):
                mimic_contact_reward = 0.2
            else:
                mimic_contact_reward = 0.0
        elif (self.mocap_data_idx <= 37):
            if (self.r_contact is False and self.l_contact is True):
                mimic_contact_reward = 0.2
            else:
                mimic_contact_reward = 0.0

        mimic_qpos_reward = 0.4 * exp(-2.0*(np.linalg.norm((target_data_qpos - qpos.flat[7:])**2).mean()))
        mimic_qvel_reward = 0.00 * exp(-0.1*(np.linalg.norm(target_data_qvel - qvel.flat[6:])**2))
        mimic_body_reward = 0.2 * exp(-10*(np.linalg.norm(Tar_Body - qpos.flat[0:3])**2)) 
        mimic_body_orientation_reward = 0.1 * exp(-200*baseQuatError)
        mimic_body_vel_reward = 0.1*exp(-10*(np.linalg.norm(target_data_body_vel - qvel.flat[0:3])**2)) # 
        reward = mimic_qpos_reward + mimic_qvel_reward + mimic_body_orientation_reward + mimic_body_reward + mimic_body_vel_reward + mimic_contact_reward

        if not done_by_contact:
            self.epi_len += 1
            self.epi_reward += reward
            return self._get_obs(), reward, done_by_contact, dict(specific_reward=dict(mimic_qpos_reward=mimic_qpos_reward, mimic_qvel_reward=mimic_qvel_reward, mimic_body_orientation_reward= mimic_body_orientation_reward, mimic_body_reward=mimic_body_reward, mimic_body_vel_reward=mimic_body_vel_reward, mimic_contact_reward=mimic_contact_reward))
        else:
            mimic_qpos_reward = 0.0
            mimic_qvel_reward = 0.0
            mimic_body_orientation_reward = 0.0
            mimic_body_reward = 0.0
            mimic_body_vel_reward = 0.0
            reward = 0.0
            return_epi_len = self.epi_len
            return_epi_reward = self.epi_reward
            return self._get_obs(), reward, done_by_contact, dict(episode=dict(r=return_epi_reward, l=return_epi_len), specific_reward=dict(mimic_qpos_reward=mimic_qpos_reward, mimic_qvel_reward=mimic_qvel_reward, mimic_body_orientation_reward= mimic_body_orientation_reward, mimic_body_reward=mimic_body_reward,mimic_body_vel_reward=mimic_body_vel_reward, mimic_contact_reward=mimic_contact_reward))

        

    def reset_model(self):
        self.time = 0.0
        self.epi_len = 0
        self.epi_reward = 0
        self.init_mocap_data_idx = np.random.randint(low=0, high=37)
        next_idx = self.init_mocap_data_idx + 1
        mocap_cycle_dt = 0.033332
        quat_desired = np.zeros(4)
        quat_desired = np.array([1,0,0,0]) #self.mocap_data[self.init_mocap_data_idx,4:8]
        self.cycle_init_root_pos = self.sim.data.qpos[0:3].copy()

        q_desired = self.mocap_data[self.init_mocap_data_idx,8:8+len(self.action_space.sample())]
        qvel_desired = (self.mocap_data[next_idx,8:8+len(self.action_space.sample())] - self.mocap_data[self.init_mocap_data_idx,8:8+len(self.action_space.sample())]) / mocap_cycle_dt
        target_data_body_vel = (self.mocap_data[next_idx,1:4] - self.mocap_data[self.init_mocap_data_idx,1:4]) / mocap_cycle_dt

        self.set_state(
            self.init_qpos + np.concatenate((np.zeros(3), -self.init_qpos[3:7] + quat_desired, q_desired)),
            self.init_qvel + np.concatenate((target_data_body_vel, np.zeros(3),qvel_desired)), 
        )        
        # + np.concatenate(((self.mocap_data[next_idx,1:4] - self.mocap_data[self.init_mocap_data_idx,1:4])/mocap_cycle_dt, np.zeros(3+len(self.action_space.sample()))))
        #
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
