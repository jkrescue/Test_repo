# 2022-11-19 WG 主函数
import os
import json
#import torch
import pickle
import paddle
import configparser
import functions as myFunc  # from DeepCFD
from paddle.distributed import fleet
from models.UNetEx import UNetEx  # from DeepCFD
from train_functions import train_model
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

def after_epoch(scope):
    train_loss_curve.append(scope["train_loss"])
    test_loss_curve.append(scope["val_loss"])
    train_mse_curve.append(scope["train_metrics"]["mse"])
    test_mse_curve.append(scope["val_metrics"]["mse"])
    train_ux_curve.append(scope["train_metrics"]["ux"])
    test_ux_curve.append(scope["val_metrics"]["ux"])
    train_uy_curve.append(scope["train_metrics"]["uy"])
    test_uy_curve.append(scope["val_metrics"]["uy"])
    train_p_curve.append(scope["train_metrics"]["p"])
    test_p_curve.append(scope["val_metrics"]["p"])


def loss_func(model, batch):
    x, y = batch
    output = model(x)
    lossu = ((output[:, 0, :, :] - y[:, 0, :, :]) ** 2).reshape((output.shape[0], 1, output.shape[2], output.shape[3]))
    lossv = ((output[:, 1, :, :] - y[:, 1, :, :]) ** 2).reshape((output.shape[0], 1, output.shape[2], output.shape[3]))
    lossp = paddle.abs((output[:, 2, :, :] - y[:, 2, :, :])).reshape(
        (output.shape[0], 1, output.shape[2], output.shape[3]))
    loss = (lossu + lossv + lossp) / channelsWeights
    return paddle.sum(loss), output


if __name__ == "__main__":
    fleet.init(is_collective=True)  # 分布式训练需要用到
    gz = GraphvizOutput(output_file=r'./TopologyGraph.png')
    with PyCallGraph(output=gz):
        # 1.Read data
        # The input information on the [geometry] of 981 channel flow samples
        # x = [ Ns, Nc, Nx, Ny]
        # Ns = 981  : Number of Samples (We have 981 geometry)
        # Nc = 3    : Number of Channels
        #             Channel 1 -> SDF calculated from the obstacle's surface
        #             Channel 2 -> multi-label flow region
        #             Channel 3 -> SDF from the top/bottom surfaces
        # Nx = 172  : Number of Elements in x
        # Ny = 79   : Number of Elements in y
        x = pickle.load(open("./data/dataX.pkl", "rb"))
        # Ground-truth CFD solution for the velocity (Ux and Uy) and the pressure (p) fields using the [simpleFOAM solver]
        y = pickle.load(open("./data/dataY.pkl", "rb"))
        # 设定种子，便于复现
        paddle.seed(999)

        # 2.Read parameters
        configPsr = configparser.ConfigParser()
        configPsr.read("./hyperParameters.txt")

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

        # 3.Data to Tensor
        x = paddle.to_tensor(x, dtype="float32", place=None, stop_gradient=True)
        y = paddle.to_tensor(y, dtype="float32", place=None, stop_gradient=True)
        y_trans = paddle.transpose(y, perm=[0, 2, 3, 1])
        channelsWeights = paddle.reshape(
            paddle.sqrt(paddle.mean(paddle.transpose(y, perm=[0, 2, 3, 1]).reshape((981 * 172 * 79, 3)) ** 2, axis=0)),
            shape=[1, -1, 1, 1])
        print("%1.3f", channelsWeights)  # check here! [0.115, 0.017, 0.013]

        simDir = "./data/"
        if not os.path.exists(simDir):
            os.makedirs(simDir)

        # 4.Splitting dataset into [70% train] and [30% test]
        trainData, testData = myFunc.split_tensors(x, y, ratio=0.7)
        train_dataset, test_dataset = \
            paddle.io.TensorDataset([trainData[0], trainData[1]]), paddle.io.TensorDataset([testData[0], testData[1]])

        # 5.Model
        model = fleet.distributed_model(
            UNetEx(3, 3, filters=filters, kernel_size=kernelSize, batch_norm=batchNorm, weight_norm=weightNorm))
        test_x, test_y = test_dataset[:]

        # 6.Optimizer
        optimizer = fleet.distributed_optimizer(
            paddle.optimizer.AdamW(learning_rate=learningRate, parameters=model.parameters(),
                                   weight_decay=weightDecay))

        # 7.Initiate vars for the Record of Losses\MSE\Velocity and Pressure
        config = {}
        train_loss_curve = []
        test_loss_curve = []
        train_mse_curve = []
        test_mse_curve = []
        train_ux_curve = []
        test_ux_curve = []
        train_uy_curve = []
        test_uy_curve = []
        train_p_curve = []
        test_p_curve = []

        # 8.Training
        DeepCFD, train_metrics, train_loss, test_metrics, test_loss = \
            train_model(model, loss_func, train_dataset, test_dataset, optimizer, epochs=epochs, batch_size=batchSize, device=0,
                        m_mse_name="Total MSE",
                        m_mse_on_batch=lambda scope: float(paddle.sum((scope["output"] - scope["batch"][1]) ** 2)),
                        m_mse_on_epoch=lambda scope: sum(scope["list"]) / len(scope["dataset"]),
                        m_ux_name="Ux MSE",
                        m_ux_on_batch=lambda scope: float(paddle.sum((scope["output"][:, 0, :, :] - scope["batch"][1][:, 0, :, :]) ** 2)),
                        m_ux_on_epoch=lambda scope: sum(scope["list"]) / len(scope["dataset"]),
                        m_uy_name="Uy MSE",
                        m_uy_on_batch=lambda scope: float(paddle.sum((scope["output"][:, 1, :, :] - scope["batch"][1][:, 1, :, :]) ** 2)),
                        m_uy_on_epoch=lambda scope: sum(scope["list"]) / len(scope["dataset"]),
                        m_p_name="p MSE",
                        m_p_on_batch=lambda scope: float(paddle.sum((scope["output"][:, 2, :, :] - scope["batch"][1][:, 2, :, :]) ** 2)),
                        m_p_on_epoch=lambda scope: sum(scope["list"]) / len(scope["dataset"]),
                        patience=25, after_epoch=after_epoch)

        # 9. Record All
        metrics = {"train_metrics": train_metrics, "train_loss": train_loss, "test_metrics": test_metrics,
                   "test_loss": test_loss}
        curves = {"train_loss_curve": train_loss_curve, "test_loss_curve": test_loss_curve,
                  "train_mse_curve": train_mse_curve, "test_mse_curve": test_mse_curve, "train_ux_curve": train_ux_curve,
                  "test_ux_curve": test_ux_curve, "train_uy_curve": train_uy_curve, "test_uy_curve": test_uy_curve,
                  "train_p_curve": train_p_curve, "test_p_curve": test_p_curve}
        config["metrics"] = metrics
        config["curves"] = curves
        with open(simDir + "results.json", "w") as file:
            json.dump(config, file)
        out = DeepCFD(test_x[:10])  # scope["best_model"]
        error = paddle.abs(out.cpu() - test_y[:10].cpu())
        myFunc.visualize(test_y[:10].cpu().detach().numpy(), out[:10].cpu().detach().numpy(),
                         error[:10].cpu().detach().numpy(), 0)
