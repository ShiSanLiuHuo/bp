# 基于numpy的BP网络

import pandas as pd
import numpy as np
import paddle

# 超参数设置
# 最大训练轮数
epoch = 2000
# 学习率
eta = 0.1
# 训练误差上界
errlimit = 0.001
# 数据集路径
datapath = "watermelon_bp.csv"
# 建立数据集替换的字典，以便将中文转换成数值类型
datadicX = {'蜷缩': 0, '稍蜷': 1, '硬挺': 2,
            '凹陷': 0, '稍凹': 1, '平坦': 2}
datadixY = {'是': [1, 0], '否': [0, 1]}

# 数据集处理类
class DataLoad:
    def __init__(self, dpath):
        # 读入数据集
        self.data = pd.read_csv(dpath)
        # 将读入的数据集转换成列表
        self.dlstX = self.data.iloc[:, 1:3].values.tolist()
        self.dlstY = self.data.iloc[:, 3].values.tolist()
        # print(self.dlstX)
        # print(self.dlstY)

    # 根据字典将列表中的字符替换为数值，并转换为ndarray
    def get_data(self):
        self.klstX = list(datadicX.keys())
        self.klstY = list(datadixY.keys())
        for i in range(len(self.dlstX)):
            for j in range(len(self.dlstX[i])):
                if self.dlstX[i][j] in self.klstX:
                    self.dlstX[i][j] = datadicX[self.dlstX[i][j]]
        for i in range(len(self.dlstY)):
            if self.dlstY[i] in self.klstY:
                self.dlstY[i] = datadixY[self.dlstY[i]]
        self.dataX = np.array(self.dlstX).astype(np.float32)
        self.dataY = np.array(self.dlstY).astype(np.float32)
        # print(self.dataX)
        # print(self.dataY)

        return self.dataX, self.dataY
    
# 根据西瓜书图5.9建立神经网络模型类
def sigmoid(x):
    return 1/(1+np.exp(-x))

class neuron: 
    def __init__(self, x_dimension, learning_rate, israndom=True):
        if israndom:
            # 随机初始化权重
            # 连接权在-2到2之间 阈值在0到2之间
            self.weight = np.random.random(x_dimension) * 4 - 2
            self.weight = np.append(self.weight, np.random.random(1) * 2)
        else:
            self.weight = np.ones(x_dimension + 1)
        self.learning_rate = learning_rate
        
    def setLearningRate(self, learning_rate):
        self.learning_rate = learning_rate 
    
    def getWeight(self):
        return self.weight
    
    def setWeight(self, weight):
        self.weight = weight
        
    def output(self, x):
        x = np.append(x, [-1])
        x = np.array(x)
        y = sigmoid(np.sum(x * self.weight))
        return y
    
class BPModel:
    # d_in 输入层维度 q_mid 中间层维度 l_out 输出层维度 eta 学习率
    def __init__(self, d_in, q_mid, l_out, eta):
        self.hidden_layer = [neuron(d_in, eta) for i in range(q_mid)]
        self.output_layer = [neuron(q_mid, eta) for i in range(l_out)]
        self.input_size = d_in
        self.hidden_size = q_mid
        self.output_size = l_out
        self.learning_rate = eta
        self.errall = 1
        self.input_data = None
        self.output_data = None
        
    def feedforward(self, x):
        b = []
        for i in self.hidden_layer:
            b.append(i.output(x))
        b = np.array(b)
        y = []
        for i in self.output_layer:
            y.append(i.output(b))
        y = np.array(y)
        return y
        
    def backpropagation(self, x, y):
        # 正向传播
        b = []
        for i in self.hidden_layer:
            b.append(i.output(x))
        b = np.array(b)
        
        y_hat = []
        for i in self.output_layer:
            y_hat.append(i.output(b))
        y_hat = np.array(y_hat)
        
        # 计算误差
        self.errall = np.sum((y - y_hat)**2) / 2
        
        # 反向传播
        # 计算输出层的梯度项 g
        g = y_hat * (1 - y_hat) * (y - y_hat)
        
        # 更新输出层权重
        for j in range(self.output_size):
            current_weight = self.output_layer[j].getWeight()
            input_to_output = np.append(b, [-1])
            updated_weight = current_weight + self.learning_rate * g[j] * input_to_output
            self.output_layer[j].setWeight(updated_weight)
        
        # 计算隐藏层的梯度项 e
        # 收集输出层权重（不包括阈值）
        w_output = np.zeros((self.output_size, self.hidden_size))
        for j in range(self.output_size):
            w_output[j, :] = self.output_layer[j].getWeight()[:-1]
        
        # 计算隐藏层误差项
        e = np.zeros(self.hidden_size)
        for h in range(self.hidden_size):
            sum_w_g = 0
            for j in range(self.output_size):
                sum_w_g += w_output[j, h] * g[j]
            e[h] = b[h] * (1 - b[h]) * sum_w_g
        
        # 更新隐藏层权重
        for h in range(self.hidden_size):
            current_weight = self.hidden_layer[h].getWeight()
            input_to_hidden = np.append(x, [-1])
            updated_weight = current_weight + self.learning_rate * e[h] * input_to_hidden
            self.hidden_layer[h].setWeight(updated_weight)
            
    def set_data(self, input_data, output_data):
        self.input_data = np.array(input_data)
        self.output_data = np.array(output_data)

    def compt(self):
        """计算并更新所有权重"""
        # 对每个样本进行反向传播
        for i in range(len(self.input_data)):
            self.backpropagation(self.input_data[i], self.output_data[i])
        
        # 收集所有权重
        self.w = []
        self.v = []
        
        # 收集隐藏层权重（包括阈值）
        for i in range(self.hidden_size):
            weights = self.hidden_layer[i].getWeight()
            for w in weights:
                self.w.append(w)
        
        # 收集输出层权重（包括阈值）
        for i in range(self.output_size):
            weights = self.output_layer[i].getWeight()
            for v in weights:
                self.v.append(v)
        
        return np.array(self.w), np.array(self.v)
    
# 可以根据下述参考代码编写上述的神经网络模型类
data = DataLoad(datapath)
dataX, dataY = data.get_data()
# print(dataX)
# print(dataY)

model = BPModel(d_in=2, q_mid=2, l_out=2, eta=eta)
model.set_data(dataX, dataY)
for i in range(epoch):
    w, v = model.compt()

    if i % 100 == 0:
        print(i, '\n', w, '\n', v,'\n',model.errall)
        print("-------------------------------\nepoch = ", i)
    if model.errall < errlimit:
        print("Done !")
        print("i = ", i)
        print("w = ", w)
        print("v = ", v)
        break