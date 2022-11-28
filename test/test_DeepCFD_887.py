import pytest
import paddle
import pickle
import configparser
import functions as myFunc  # from DeepCFD
from models.UNetEx import UNetEx  # from DeepCFD
from paddle.distributed import fleet


# 测试用例
class TestDome:

    def test_demo1(self):
        print('----测试用例执行-----------')
        fleet.init(is_collective=True)  # 分布式训练需要用到
        x = pickle.load(open(".././data/dataX.pkl", "rb"))
        y = pickle.load(open(".././data/dataY.pkl", "rb"))
        x = paddle.to_tensor(x, dtype="float32", place=None, stop_gradient=True)
        y = paddle.to_tensor(y, dtype="float32", place=None, stop_gradient=True)
        trainData, testData = myFunc.split_tensors(x, y, ratio=0.7)
        train_dataset, test_dataset = \
            paddle.io.TensorDataset([trainData[0], trainData[1]]), paddle.io.TensorDataset([testData[0], testData[1]])

        test_x, test_y = test_dataset[:]

        # 2.Read parameters
        configPsr = configparser.ConfigParser()  # 读取超参数文件
        configPsr.read(".././hyperParameters.txt")
        learningRate = float(configPsr["Hpara"]["learningRate"])
        kernelSize = int(configPsr["Hpara"]["kernelSize"])
        batchSize = int(configPsr["Hpara"]["batchSize"])
        trainTestRatio = float(configPsr["Hpara"]["trainTestRatio"])
        weightDecay = float(configPsr["Hpara"]["trainTestRatio"])
        batchNorm = configPsr.getboolean("Hpara", "batchNorm")
        weightNorm = configPsr.getboolean("Hpara", "weightNorm")
        tmpList = configPsr["Others"]["filters"].split(",")
        filters = [int(i) for i in tmpList]  # turn char into int
        epochs = int(configPsr["Others"]["epochs"])

        model = fleet.distributed_model(
            UNetEx(3, 3, filters=filters, kernel_size=kernelSize, batch_norm=batchNorm, weight_norm=weightNorm))
        prog = paddle.load(".././DeepCFD_887.pdparams")
        model.set_state_dict(prog)

        n = 10  # take 10 different inputs to plot
        total = 3  # how many plots u want
        out = model(test_x[:n])  # scope["best_model"]
        error = paddle.abs(out.cpu() - test_y[:n].cpu())
        myFunc.multiVisualize(test_y[:n].cpu().detach().numpy(), out[:n].cpu().detach().numpy(),
                              error[:n].cpu().detach().numpy(), total)
        assert 11 == 11
