import gym
import numpy as np
import math
import pybullet as p
from robot.resources.op3 import OP3
from robot.resources.plane import Plane
from robot.resources.sphere import sphere
from robot.resources.sphere import box
from gym.utils import seeding
from robot.resources.savereward import save
import time

class Op3Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Op3Env, self).__init__()

        '''
        # 動作空間
        # 第一個維度:['l_sho_pitch'] = [1.5,-1.5]
        # 第二個維度:['l_sho_roll'] = [1.5,-1.5]
        # 第二個維度:['l_el'] = [1.5,-1.5]
        '''

        self.action_space = gym.spaces.box.Box(
            low=np.array([-2.0,-2.0,-2.0], dtype=np.float32),
            high=np.array([2.0,2.0,2.0], dtype=np.float32))

        '''
        # 觀察空間
        # 索引[0,1,3] :手的xyz座標[-2,2]
        索引[4,5,6] :球的xyz方向[-2,2]
        '''

        self.observation_space = gym.spaces.box.Box(
            low=np.array([-2, -2, -2, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([2, 2, 2, 0.2, 0.2, 0.35], dtype=np.float32))

        self.np_random, _ = gym.utils.seeding.np_random()

        # 選擇連結方式
        # self.client = p.connect(p.DIRECT)
        self.client = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI)
        # 加速訓練
        p.setTimeStep(1/180, self.client)
        # 初始化所有東西
        self.OP3 = None
        self.sphere = None
        self.box = None
        self.done = False
        self.get_point = False
        np_resource = np.dtype([("resource", np.ubyte, 1)])
        # self.reset
        arms2point = np.array([0.16, 0.13, 0.32])
        self.arm2point_len = np.linalg.norm(arms2point)
        self.count = 0
        self.grab_counter = 0
        self.reward = 0
    def step(self, action):
        self.state += action

        self.OP3.apply_action(self.state)
        # print("State = ",self.state)
        p.stepSimulation()
        OP3_ob = self.OP3.get_observation()
        OP3_ob = list(OP3_ob)
        self.state = OP3_ob
        pos = self.OP3.getlel()
        ball_ob = self.sphere.get_pos()
        dis = abs(np.linalg.norm(np.asarray(pos) - np.asarray(ball_ob)))
        self.OP3.setCameraPicAndGetPic()
        self.reward = self.reward_fun(dis)

        # print('pos>>>', pos)
        # print('ball_pso>>>',ball_ob)
        # print("dis = ",dis)
        self.count += 1
        if(self.count == 5000):
            self.sphere.newpos()
            self.count = 0

        ball_pos = list(ball_ob)
        ob = OP3_ob + ball_pos
        ob = np.array(ob, dtype=np.float32)
        # print("reward = ", reward)
        # self.OP3.setCameraPicAndGetPic()
        info = {}
        return ob, self.reward, self.done, info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    @property
    def reset(self):
        # 初始化所有東西
        print('reward = ',self.reward)
        save(self.reward)
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8)
        self.OP3 = OP3(self.client)
        self.box = box(self.client)
        Plane(self.client)
        self.sphere = sphere(self.client)
        self.done = False
        self.OP3.resetandstart()
        # self.sphere.newpos()
        OP3_ob = self.OP3.get_observation()
        OP3_ob = list(OP3_ob)
        ball_ob = self.sphere.get_pos()
        ball_ob = np.array(ball_ob,dtype=np.float32)
        ball_ob = list(ball_ob)
        ob = OP3_ob + ball_ob
        self.state = OP3_ob
        self.grab_counter = 0
        # OP3_ob =self.np_random.uniform(low=-0.05, high=0.05, size=(3,))
        return np.array(ob,dtype=np.float32)

    def render(self, mode='human'):
        # view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7, 0, 0.05],
        #                                                   distance=.7,
        #                                                   yaw=90,
        #                                                   pitch=-70,
        #                                                   roll=0,
        #                                                   upAxisIndex=2)
        # proj_matrix = p.computeProjectionMatrixFOV(fov=60,
        #                                            aspect=float(960) / 720,
        #                                            nearVal=0.1,
        #                                            farVal=100.0)
        # (_, _, px, _, _) = p.getCameraImage(width=960,
        #                                     height=720,
        #                                     viewMatrix=view_matrix,
        #                                     projectionMatrix=proj_matrix,
        #                                     renderer=p.ER_BULLET_HARDWARE_OPENGL)
        #
        # rgb_array = np.array(px, dtype=np.uint8)
        # rgb_array = np.reshape(rgb_array, (720, 960, 4))
        #
        # rgb_array = rgb_array[:, :, :3]
        # return rgb_array
        pass

    def close(self):
        p.disconnect(self.client)

    def reward_fun(self, distance):
        t = 10
        r = -distance / self.arm2point_len
        if distance < 0.03 and (not self.done):
            r += 1.
            self.grab_counter += 1
            r += self.grab_counter
            if self.grab_counter > t:
                # r += 10.
                self.done = True
                self.count = 0
        elif distance > 0.03:
            self.grab_counter = 0
            self.done = False

        return r
