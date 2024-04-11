##### 基于PPO算法的轨迹规划
现有基于强化学习的规划算法通常只涉及空间层面上的<font color = 'red'>**路径规划**</font>，即将物理空间划分为离散网格的形式，再通过选择临近网格的方式进行规划，这种规划方式局限性很大，不能很好地指导智能体的运动与控制。

与之相对的，在时间层面上进行的<font color = 'lime'>**轨迹规划**</font>可以输出每个时刻智能体的期望速度，与后续的控制器设计过程更加契合。

下面使用PPO算法实现智能体的目标追踪轨迹规划。

强化学习建模包括状态、动作、奖励、状态转移函数等。  
为简单考虑，以智能体为原点建立坐标系。 

智能体在时刻 $t$ 的状态 $s_t$ 包括目标坐标，航向角，上一时刻的绝对速度： $$s_t=(p^{gx}_t,p^{gy}_t,\theta_t,v^{xlast}_t,v^{ylast}_t,r^{last}_t)$$ 


智能体在时刻 $t$ 采取的动作 $a_t$ 为三个维度的绝对速度： $$a_t=(v^x_t,v^y_t,r_t)$$   

智能体的状态转移函数 $f(s_t,a_t)$ 为： $$p^{gx}_{t+1}=p^{gx}_t-\Delta t \cdot v^x_t$$

$$p^{gy}_{t+1}=p^{gy}_t-\Delta t \cdot v^y_t$$

$$\theta_{t+1}=\theta_{t}+\Delta t \cdot r_t$$

$$v^{xlast}_{t+1}=v^{x}_t$$

$$v^{ylast}_{t+1}=v^{y}_t$$

$$r^{last}_{t+1}=r_t$$ 

智能体的奖励函数 $r_t = \mathbb{R}(s_t,a_t)$ 为：

$$\mathbb{R}=-\alpha d_t^2 -\beta\big(v_t \max(0,\gamma-d_t^2)\big) + \mathbf{R} - \mathbf{J}_1 - \mathbf{J}_2$$

其中， $\mathbf{R}$ 代表到达目标的奖励， $\mathbf{J}_1$ 用以衡量速度方向和航向角的偏差， $\mathbf{J}_2$ 用以衡量速度的平滑度


![图片]((https://github.com/JackWafson/trajectory-planning-with-PPO/blob/main/figure/figure1.svg))

上述建模是一个简单的目标追踪，没有考虑避障、追逃等更加复杂的问题。对这些问题的分析都需要智能体对外界的感知。假设智能体可以通过激光雷达获得外界物体以智能体为原点的相对位置和绝对速度（由雷达信号获得这些信息的具体处理过程略）。
