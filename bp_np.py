# 基于numpy的BP网络

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
# 建立数据集替换的字典，以便将中文转换成数值类型
datadicX = {'蜷缩': 0, '稍蜷': 1, '硬挺': 2,
            '凹陷': 0, '稍凹': 1, '平坦': 2}
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
        raw_x = [[row["根蒂"], row["脐部"]] for row in self.data]
        raw_y = [row["好瓜"] for row in self.data]

        data_x_list: list[list[float]] = []
        data_y_list: list[list[float]] = []

        for row in raw_x:
            mapped_row = [float(datadicX[str(value)]) for value in row]
            data_x_list.append(mapped_row)

        for value in raw_y:
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
        x = np.append(x, [-1])
        x = np.array(x)
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
        sample_error = np.sum((y - y_hat)**2) / 2
        
        # 反向传播
        # 计算输出层的梯度项 g
        g = y_hat * (1 - y_hat) * (y - y_hat)
        
        # 在更新前保存输出层权重，供隐藏层梯度计算使用
        w_output = np.zeros((self.output_size, self.hidden_size))
        for j in range(self.output_size):
            w_output[j, :] = self.output_layer[j].getWeight()[:-1]

        # 更新输出层权重
        for j in range(self.output_size):
            current_weight = self.output_layer[j].getWeight()
            input_to_output = np.append(b, [-1])
            updated_weight = current_weight + self.learning_rate * g[j] * input_to_output
            self.output_layer[j].setWeight(updated_weight)
        
        # 计算隐藏层的梯度项 e
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

        return sample_error
            
    def set_data(self, input_data: FloatArray, output_data: FloatArray) -> None:
        self.input_data = np.array(input_data)
        self.output_data = np.array(output_data)

    def compute_dataset_error(self) -> float:
        total_error = 0.0
        for i in range(len(self.input_data)):
            y_hat = self.feedforward(self.input_data[i])
            total_error += np.sum((self.output_data[i] - y_hat) ** 2) / 2
        return total_error / len(self.input_data)

    def train_one_epoch(self) -> tuple[FloatArray, FloatArray]:
        """完成一轮训练，并返回当前所有权重。"""
        # 对每个样本进行反向传播
        for i in range(len(self.input_data)):
            self.backpropagation(self.input_data[i], self.output_data[i])

        # 使用整轮更新后的最新权重重新计算训练集平均误差。
        self.errall = self.compute_dataset_error()
        self.loss_history.append(self.errall)
        
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


def save_loss_curve(loss_history: list[float], output_path: Path) -> None:
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
    data = DataLoad(datapath)
    dataX, dataY = data.get_data()

    model = BPModel(d_in=2, q_mid=2, l_out=2, eta=eta)
    model.set_data(dataX, dataY)
    for i in range(epoch):
        w, v = model.train_one_epoch()

        if i % 100 == 0:
            print(i, '\n', w, '\n', v, '\n', model.errall)
            print("-------------------------------\nepoch = ", i)
        if model.errall < errlimit:
            print("Done !")
            print("i = ", i)
            print("w = ", w)
            print("v = ", v)
            break
    else:
        print("Training stopped without reaching errlimit.")
        print("final err = ", model.errall)

    save_loss_curve(model.loss_history, loss_plot_path)
    print("loss curve saved to", loss_plot_path)


if __name__ == "__main__":
    main()
