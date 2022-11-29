import json
import unittest
from matplotlib import pyplot as plt


def mseVisualize(fourMSE):
    epochsArray = list(range(1, 1001))

    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.suptitle('Test Error(MSE)', fontsize=18)
    plt.subplot(2, 2, 1)
    plt.title('Validation Total MSE', fontsize=18)
    plt.ylabel('Total MSE', fontsize=18)
    plt.ylim([0, 10])
    plt.xlim([0, 1000])
    plt.plot(epochsArray, fourMSE["Validation Total MSE"], color='b', linewidth=1)

    plt.subplot(2, 2, 2)
    plt.title('Validation Ux MSE', fontsize=18)
    plt.ylabel('Ux MSE', fontsize=18)
    plt.ylim([0, 8])
    plt.xlim([0, 1000])
    plt.plot(epochsArray, fourMSE["Validation Ux MSE"], color='b', linewidth=1)

    plt.subplot(2, 2, 3)
    plt.title('Validation Uy MSE', fontsize=18)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Uy MSE', fontsize=18)
    plt.ylim([0, 1.5])
    plt.xlim([0, 1000])
    plt.plot(epochsArray, fourMSE["Validation Uy MSE"], color='b', linewidth=1)

    plt.subplot(2, 2, 4)
    plt.title('Validation p MSE', fontsize=18)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Uy MSE', fontsize=18)
    plt.ylim([0, 5])
    plt.xlim([0, 1000])
    plt.plot(epochsArray, fourMSE["Validation p MSE"], color='b', linewidth=1)
    plt.show()


class MyTestCase(unittest.TestCase):
    def test_something(self):
        Validation_Ux_mse_curve = [float(0.0)]*1000
        Validation_Uy_mse_curve = [float(0.0)]*1000
        Validation_P_mse_curve = [float(0.0)]*1000

        with open("../20221125014750 TrainLog.txt", "r") as file:
            counter = 0
            for lines in file.readlines():
                if "Validation Ux MSE" in str(lines):
                    rs = lines.replace('\n', '')
                    Validation_Ux_mse_curve[counter] = float(rs[21:])
                if "Validation Uy MSE" in str(lines):
                    rs = lines.replace('\n', '')
                    Validation_Uy_mse_curve[counter] = float(rs[21:])
                if "Validation p MSE" in str(lines):
                    rs = lines.replace('\n', '')
                    Validation_P_mse_curve[counter] = float(rs[20:])
                    counter = counter + 1

        with open(".././data/results.json", "r") as file:
            config = json.load(file)
        MSEdata = config["curves"]

        fourMSE = {"Validation Total MSE": MSEdata["test_mse_curve"], "Validation Ux MSE": Validation_Ux_mse_curve,
                   "Validation Uy MSE": Validation_Uy_mse_curve, "Validation p MSE": Validation_P_mse_curve}

        mseVisualize(fourMSE)

        self.assertEqual(True, True)  # add assertion here



if __name__ == '__main__':
    unittest.main()
