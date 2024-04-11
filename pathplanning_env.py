import numpy as np
import math
import random
	
π = math.pi

class pathplanning():

    def __init__(self):

        self.Δt = 0.1 # 时间步长

        self.state = None

    def step(self, action):

        Δt = self.Δt

        p_gx, p_gy, θ, v_xl, v_yl, r_l = self.state

        v_x, v_y, r = action
        
        # 轨迹规划方程
        p_gx = p_gx - Δt*v_x
        p_gy = p_gy - Δt*v_y        
        θ = θ + Δt*r
        if θ>π:
            θ = θ - 2*π
        elif θ<-π:
            θ = θ + 2*π
        
        # 计算奖励值
        reward = self.compute_reward(action)

        v_xl = v_x
        v_yl = v_y
        r_l = r

        self.state = np.array([p_gx, p_gy, θ, v_xl, v_yl, r_l])

        done = 0
        if p_gx**2 + p_gy**2 < 1:
            done = 1

        return self.state, reward, done, {}
    
    def compute_reward(self, action):

        p_gx, p_gy, θ, v_xl, v_yl, r_l = self.state
        v_x, v_y, r = action

        w1 = 10
        # w2 = 1
        w3 = 1
        w4 = 0
        w5 = 1
        # b1 = 1

        d = math.sqrt(p_gx**2+p_gy**2)
        v = math.sqrt(v_x**2+v_y**2)

        reward_1 = -w1*d #- w2*v*max(0, b1-d) # 衡量目标点与智能体的距离，暂不考虑停在目标点的问题
        
        reward_2 = -w3*((v_x-v_xl)**2 + (v_y-v_yl)**2 + (r-r_l)**2) # 衡量速度平滑度

        θ_v = math.atan2(v_y, v_x)
        θ_d = abs(θ - θ_v)
        if θ_d > π:
            θ_d = 2*π - θ_d

        reward_3 = -w4*(θ_d**2) # 衡量速度方向与航向角方向的偏差

        reward_4 = -w5*((v_x)**2 + (v_y)**2 + (r)**2) # 衡量速度大小

        reward_5 = 0 # 到达目标奖励
        
        if p_gx**2 + p_gy**2 < 1:
            reward_5 = 20000

        reward = reward_1 + reward_2 + reward_3 + reward_4 + reward_5
        
        return reward
    
    def reset(self):
        
        # 随机生成一组在半径为5的圆上的目标点坐标
        p_gx = random.uniform(-5, 5)
        p_gy = math.sqrt(25 - p_gx**2) * (random.randint(0,1)*2-1)

        θ = 0
        v_xl = 0
        v_yl = 0
        r_l = 0

        self.state = np.array([p_gx, p_gy, θ, v_xl, v_yl, r_l])

        return self.state

        