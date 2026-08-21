import numpy as np
import random
import tensorflow as tf
import os

Val = 3
random.seed(0)
def black_box_function(Para):
    ParaFile = open("C:/Users/Administrator/Desktop/PLATFORM_5G_SCLF/x64/Release/Para.txt", 'w')
    ParaFile.write(str(Para[0]*2)+' '+str(Para[1]*1)+' '+str(Para[2]/4))
    ParaFile.close()

    s0 = "C:/Users/Administrator/Desktop/PLATFORM_5G_SCLF/x64/Release/platform_5G.exe"
    os.system(s0)
    with open("C:/Users/Administrator/Desktop/PLATFORM_5G_SCLF/x64/Release/FER.txt", 'r') as ResultFile:
        for line in ResultFile:
            Result = [float(e) for e in line.split()]
    ResultFile.close()

    return Result[1], Result[0]<= 0.108030 #1.6dB:0.0125*1.15  0.8dB 0.380228*1.15  1.8dB 0.003241*1.15

class BlackBoxEnv:
    def __init__(self):
        self.aa = np.array([16,3,0.75]).astype(float)/np.array([2,1,0.25]).astype(float)
        self.current_state = np.array(self.aa)  # *2 *1 /4
        self.action_space = Val*2  # 假设每个参数有2个选择（增加或减少）
        self.best = np.array(self.aa)
        self.bestObj = 1e4

    def reset(self):
        self.current_state = np.array(self.aa)
        return self.current_state

    def step(self, action):
        # 这里将action转化为对参数的增减
        param_index = action // 2  # 选择哪个参数
        change = 1 if action % 2 == 0 else -1  # 增加或减少1
        self.current_state[param_index] += change
        
        # 调用黑盒计算目标和约束
        target_value, constraint_value = black_box_function(self.current_state)
        if self.bestObj > target_value and constraint_value == True:
            self.bestObj = target_value
            self.best = np.array(self.current_state)

        print(self.current_state, target_value, constraint_value, self.best, self.bestObj)

        LogFile = open("log.txt", 'a')
        for value in [self.current_state, target_value, constraint_value]:
            LogFile.write(str(value)+' ')
        LogFile.write('\n')
        LogFile.close()
        
        # 计算奖励
        done = False
        if constraint_value < 0:  # 约束条件满足
            reward = -target_value  # 奖励为负目标函数值
        else:
            reward = -1000  # 惩罚

        return self.current_state, reward, done, {"target_value": target_value, "constraint_value": constraint_value}


class DQNAgent:
    def __init__(self):
        self.memory = []
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=Val, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(Val*2, activation='linear')  # 输出Val*2个动作的Q值
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(Val*2)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.max(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

# 进行训练
env = BlackBoxEnv()
agent = DQNAgent()

for e in range(1000):  # 训练轮次
    state = env.reset()
    state = np.reshape(state, [1, Val])
    for time in range(30):  # 每个episode的最大步数
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, Val])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    agent.replay(32)  # 进行经验回放
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay
