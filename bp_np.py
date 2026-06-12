# 基于numpy的BP网络
# 这个实现对应一个三层前馈神经网络：输入层 -> 隐藏层 -> 输出层。
# 训练时使用标准 BP（误差反向传播）算法，按样本逐个更新权重。

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)

FloatArray = np.ndarray

# 超参数设置
# 最大训练轮数
epoch = 2000
# 学习率
eta = 0.2
# 训练误差上界
errlimit = 0.001
# 数据集路径
datapath = Path(__file__).with_name("watermelon_bp.csv")
# loss 曲线输出路径
loss_plot_path = Path(__file__).with_name("bp_loss_curve.png")
# 建立数据集替换字典，把离散中文属性编码成数值。
# 这里采用的是最直接的人工编码方式，便于送入神经网络做数值计算。
datadicX = {'蜷缩': 0, '稍蜷': 1, '硬挺': 2,
            '凹陷': 0, '稍凹': 1, '平坦': 2}
# 输出采用 one-hot 编码：
# “是” -> [1, 0]， “否” -> [0, 1]
# 这样输出层两个神经元分别表示属于两类的程度。
datadixY = {'是': [1, 0], '否': [0, 1]}

# 数据集处理类
class DataLoad:
    def __init__(self, dpath: Path):
        # 读入数据集
        with dpath.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            self.data = list(reader)

    # 根据字典将列表中的字符替换为数值，并转换为ndarray
    def get_data(self) -> tuple[FloatArray, FloatArray]:
        # 从每一行中取出输入特征和标签。
        raw_x = [[row["根蒂"], row["脐部"]] for row in self.data]
        raw_y = [row["好瓜"] for row in self.data]

        data_x_list: list[list[float]] = []
        data_y_list: list[list[float]] = []

        for row in raw_x:
            # 把每个离散特征映射为浮点数，便于后续矩阵/向量计算。
            mapped_row = [float(datadicX[str(value)]) for value in row]
            data_x_list.append(mapped_row)

        for value in raw_y:
            # 标签映射为 one-hot 向量，对应输出层两个节点的目标值。
            data_y_list.append([float(item) for item in datadixY[str(value)]])

        data_x = np.array(data_x_list, dtype=np.float32)
        data_y = np.array(data_y_list, dtype=np.float32)

        return data_x, data_y
    
# 根据西瓜书图5.9建立神经网络模型类
def sigmoid(x):
    return 1/(1+np.exp(-x))

class Neuron:
    def __init__(self, x_dimension: int, learning_rate: float, israndom: bool = True):
        if israndom:
            # 随机初始化权重
            # 连接权在-2到2之间 阈值在0到2之间
            # 随机初始化的目的是打破对称性，避免所有神经元学成完全一样
            self.weight = np.random.random(x_dimension) * 4 - 2
            self.weight = np.append(self.weight, np.random.random(1) * 2)
        else:
            self.weight = np.ones(x_dimension + 1)
        self.learning_rate = learning_rate
        
    def setLearningRate(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate 
    
    def getWeight(self) -> FloatArray:
        return self.weight
    
    def setWeight(self, weight: FloatArray) -> None:
        self.weight = weight
        
    def output(self, x: FloatArray) -> float:
        # 追加一个固定输入 -1，用最后一位权重表示阈值项。
        # 把“加阈值/减偏置”统一写成一次向量点积。
        x = np.append(x, [-1])
        x = np.array(x)
        # 神经元输出 = sigmoid(加权输入和)
        y = sigmoid(np.sum(x * self.weight))
        return float(y)
    
class BPModel:
    # d_in 输入层维度 q_mid 中间层维度 l_out 输出层维度 eta 学习率
    def __init__(self, d_in: int, q_mid: int, l_out: int, eta: float):
        self.hidden_layer = [Neuron(d_in, eta) for i in range(q_mid)]
        self.output_layer = [Neuron(q_mid, eta) for i in range(l_out)]
        self.input_size = d_in
        self.hidden_size = q_mid
        self.output_size = l_out
        self.learning_rate = eta
        self.errall = 1.0
        self.input_data: FloatArray = np.empty((0, d_in), dtype=np.float32)
        self.output_data: FloatArray = np.empty((0, l_out), dtype=np.float32)
        self.loss_history: list[float] = []
        
    def feedforward(self, x: FloatArray) -> FloatArray:
        # 前向传播分两步：
        # 1. 输入层 -> 隐藏层，得到隐藏层输出 b
        # 2. 隐藏层 -> 输出层，得到网络预测 y
        b = []
        for i in self.hidden_layer:
            b.append(i.output(x))
        b = np.array(b)

        y = []
        for i in self.output_layer:
            y.append(i.output(b))
        y = np.array(y)
        return y
        
    def backpropagation(self, x: FloatArray, y: FloatArray) -> float:
        # 对单个样本执行一次“前向传播 + 反向传播 + 权重更新”。
        # 这里是随机梯度下降（SGD）的写法，因为每次只用一个样本更新参数。

        # 正向传播：先算隐藏层输出 b，再算最终预测 y_hat。
        b = []
        for i in self.hidden_layer:
            b.append(i.output(x))
        b = np.array(b)
        
        y_hat = []
        for i in self.output_layer:
            y_hat.append(i.output(b))
        y_hat = np.array(y_hat)
        
        # 采用平方误差：E_k = 1/2 * sum((y - y_hat)^2)
        sample_error = np.sum((y - y_hat)**2) / 2
        
        # 反向传播
        # 输出层梯度项 g：
        # g_j = y_hat_j * (1 - y_hat_j) * (y_j - y_hat_j)
        # 前半部分来自 sigmoid 的导数，后半部分来自平方误差对输出的导数。
        g = y_hat * (1 - y_hat) * (y - y_hat)
        
        # 在更新输出层之前，先保存旧权重。
        # 原因是隐藏层误差项要用“输出层到隐藏层”的旧连接权来回传误差信号。
        w_output = np.zeros((self.output_size, self.hidden_size))
        for j in range(self.output_size):
            w_output[j, :] = self.output_layer[j].getWeight()[:-1]

        # 更新输出层权重：
        # Delta_w = eta * g_j * b
        # 其中 b 是隐藏层输出，也就是输出层当前神经元的输入。
        for j in range(self.output_size):
            current_weight = self.output_layer[j].getWeight()
            input_to_output = np.append(b, [-1])
            updated_weight = current_weight + self.learning_rate * g[j] * input_to_output
            self.output_layer[j].setWeight(updated_weight)
        
        # 隐藏层误差项 e_h：
        # e_h = b_h * (1 - b_h) * sum(w_hj * g_j)
        # 含义是：隐藏层节点 h 的误差，等于它对后续所有输出节点造成影响的总和，
        # 再乘上它自身激活函数的导数。
        e = np.zeros(self.hidden_size)
        for h in range(self.hidden_size):
            sum_w_g = 0
            for j in range(self.output_size):
                sum_w_g += w_output[j, h] * g[j]
            e[h] = b[h] * (1 - b[h]) * sum_w_g
        
        # 更新隐藏层权重：
        # Delta_v = eta * e_h * x
        # 这里 x 是原始输入向量，因此隐藏层权重直接根据输入和隐藏层误差项调整。
        for h in range(self.hidden_size):
            current_weight = self.hidden_layer[h].getWeight()
            input_to_hidden = np.append(x, [-1])
            updated_weight = current_weight + self.learning_rate * e[h] * input_to_hidden
            self.hidden_layer[h].setWeight(updated_weight)

        return sample_error
            
    def set_data(self, input_data: FloatArray, output_data: FloatArray) -> None:
        # 保存训练集，后续每一轮训练都会遍历这些样本。
        self.input_data = np.array(input_data)
        self.output_data = np.array(output_data)

    def compute_dataset_error(self) -> float:
        # 计算整个训练集上的平均误差，用于观察整体收敛情况。
        # 注意：它不参与本轮权重更新，只用于记录训练效果。
        total_error = 0.0
        for i in range(len(self.input_data)):
            y_hat = self.feedforward(self.input_data[i])
            total_error += np.sum((self.output_data[i] - y_hat) ** 2) / 2
        return total_error / len(self.input_data)

    def train_one_epoch(self) -> tuple[FloatArray, FloatArray]:
        """完成一轮训练，并返回当前所有权重。"""
        # 一个 epoch 表示把整个训练集完整训练一遍。
        # 这里采用逐样本更新，所以同一轮中后面的样本会使用前面样本更新后的新权重。
        for i in range(len(self.input_data)):
            self.backpropagation(self.input_data[i], self.output_data[i])

        # 使用整轮更新后的最新权重重新计算训练集平均误差。
        self.errall = self.compute_dataset_error()
        self.loss_history.append(self.errall)
        
        # 收集所有权重，主要是为了训练过程中打印出来观察参数变化。
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


def save_loss_curve(loss_history: list[float], output_path: Path) -> None:
    # 将每轮训练得到的平均误差画成曲线。
    # 以及学习率是否可能过大或过小。
    if not loss_history:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(loss_history)), loss_history, color="tab:blue", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("BP Training Loss Curve")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    # 读取并编码数据集
    data = DataLoad(datapath)
    dataX, dataY = data.get_data()

    # 初始化 BP 网络结构
    # 当前结构为 2-2-2：2 个输入节点，2 个隐藏层节点，2 个输出节点。
    model = BPModel(d_in=2, q_mid=2, l_out=2, eta=eta)
    model.set_data(dataX, dataY)

    # 循环训练，直到达到最大轮数或误差降到阈值以下
    for i in range(epoch):
        w, v = model.train_one_epoch()

        if i % 100 == 0:
            # 每隔 100 轮打印一次参数和误差，便于查看训练过程。
            print(i, '\n', w, '\n', v, '\n', model.errall)
            print("-------------------------------\nepoch = ", i)
        if model.errall < errlimit:
            # 当整体误差低于设定阈值时，认为模型已经基本收敛。
            print("Done !")
            print("i = ", i)
            print("w = ", w)
            print("v = ", v)
            break
    else:
        # 最大轮数时仍未满足误差要求。
        print("Training stopped without reaching errlimit.")
        print("final err = ", model.errall)

    # 保存 loss 曲线
    save_loss_curve(model.loss_history, loss_plot_path)
    print("loss curve saved to", loss_plot_path)


if __name__ == "__main__":
    main()
