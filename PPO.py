import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathplanning_env import pathplanning
import time
import matplotlib.pyplot as plt
import os, sys

os.chdir(sys.path[0])

LR_v = 1e-5
LR_p = 1e-5
K_epoch = 8
GAMMA = 0.99
LAMBDA = 0.95
CLIP = 0.2

env = pathplanning()

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        self.μ = nn.Linear(256, 3)
        self.σ = nn.Linear(256, 3)
        self.optim = torch.optim.Adam(self.parameters(), lr=LR_p)
 
    def forward(self, x):
        x = self.net(x)
        μ = torch.tanh(self.μ(x)) * 1
        σ = F.softplus(self.σ(x)) + 1e-7
        return μ, σ
    
class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.optim = torch.optim.Adam(self.parameters(), lr=LR_v)
 
    def forward(self, x):
        x = self.net(x)
        return x
    
class Agent(object):
    def __init__(self):
        self.v = Value()        
        self.p = Policy()
        self.old_p = Policy()        #旧策略网络
        self.old_v = Value()         #旧价值网络    用于计算上次更新与下次更新的差别

        self.data = []               #用于存储经验
        self.step = 0
    
    def choose_action(self, s):
        with torch.no_grad():
            μ, σ = self.old_p(s)
            actions = []
            for i in range(3):
                distribution = torch.distributions.Normal(μ[i], σ[i])
                action = distribution.sample()
                actions.append(action.item())
        return actions
 
    def push_data(self, transitions):
        self.data.append(transitions)
 
    def sample(self):
        l_s, l_a, l_r, l_s_, l_done = [], [], [], [], []
        for item in self.data:
            s, a, r, s_, done = item
            l_s.append(torch.tensor([s], dtype=torch.float))
            l_a.append(torch.tensor([a], dtype=torch.float))
            l_r.append(torch.tensor([[r]], dtype=torch.float))
            l_s_.append(torch.tensor([s_], dtype=torch.float))
            l_done.append(torch.tensor([[done]], dtype=torch.float))
        s = torch.cat(l_s, dim=0)
        a = torch.cat(l_a, dim=0)
        r = torch.cat(l_r, dim=0)
        s_ = torch.cat(l_s_, dim=0)
        done = torch.cat(l_done, dim=0)
        self.data = []
        return s, a, r, s_, done
 
    def update(self):
        self.step += 1
        s, a, r, s_, done = self.sample()
        for _ in range(K_epoch):
            with torch.no_grad():
                
                '''用于计算价值网络loss'''
                td_target = r + GAMMA * self.old_v(s_) * (1 - done)
                
                
                '''用于计算策略网络loss'''
                μ, σ = self.old_p(s)
                log_prob_old = 0
                for i in range(3):
                    μ_i = μ[:, i].unsqueeze(1)
                    σ_i = σ[:, i].unsqueeze(1)
                    old_dist_i = torch.distributions.Normal(μ_i, σ_i)
                    a_i = a[:, i].unsqueeze(1)
                    log_prob_old += old_dist_i.log_prob(a_i)

                td_error = r + GAMMA * self.v(s_) * (1 - done) - self.v(s)
                td_error = td_error.detach().numpy()
                A = []
                adv = 0.0
                for td in td_error[::-1]:
                    adv = adv * GAMMA * LAMBDA + td[0]
                    A.append(adv)
                A.reverse()
                A = torch.tensor(A, dtype=torch.float).reshape(-1, 1)              

            μ, σ = self.p(s)
            log_prob_new = 0
            for i in range(3):
                μ_i = μ[:, i].unsqueeze(1)
                σ_i = σ[:, i].unsqueeze(1)
                new_dist_i = torch.distributions.Normal(μ_i, σ_i)
                a_i = a[:, i].unsqueeze(1)
                log_prob_new += new_dist_i.log_prob(a_i)

   
            ratio = torch.exp(log_prob_new - log_prob_old)

            L1 = ratio * A
            L2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * A
            loss_p = -torch.min(L1, L2).mean()
            self.p.optim.zero_grad()
            loss_p.backward()
            self.p.optim.step()
 
            loss_v = F.huber_loss(td_target.detach(), self.v(s))
            self.v.optim.zero_grad()
            loss_v.backward()
            self.v.optim.step()

        self.old_p.load_state_dict(self.p.state_dict())
        self.old_v.load_state_dict(self.v.state_dict())
 
    def save(self):
        torch.save(self.p.state_dict(), r'.\model\p.pth')
        torch.save(self.v.state_dict(), r'.\model\v.pth')
        # print('...save model...')
 
    def load(self):
        try:
            self.p.load_state_dict(torch.load(r'.\model\p.pth'))
            self.v.load_state_dict(torch.load(r'.\model\v.pth'))
            # print('...load...')
        except:
            pass