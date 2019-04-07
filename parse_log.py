import os
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def parse_log_file(log_file):
    parser = {"train": {"epoch": [], "loss": []},
              "val": {"epoch": [], "loss": [], "acc": []}}
    with open(log_file) as f:
        is_train_epoch= True
        for log_line in f.readlines():
            if "epoch:" in log_line:
                epoch = int(log_line.strip().split()[-1])
                if "Training" in log_line:
                    parser["train"]["epoch"].append(epoch)
                    is_train_epoch = True
                else:
                    parser["val"]["epoch"].append(epoch)
                    is_train_epoch = False
            if "mean_loss:" in log_line:
                loss = float(log_line.strip().split(":")[-1])
                if is_train_epoch:
                    parser["train"]["loss"].append(loss)
                else:
                    parser["val"]["loss"].append(loss)
            if "Top1:" in log_line:
                acc = float(log_line.strip().split(":")[-1][:-1])
                parser["val"]["acc"].append(acc)

    return parser


if __name__ == "__main__":
    parser = parse_log_file("train_lstm_last.log")
    # parser = parse_log_file("train_lstm.log")
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    plt.sca(ax1)
    plt.plot(parser["train"]["epoch"], parser["train"]["loss"], label="training loss")
    plt.plot(parser["val"]["epoch"], parser["val"]["loss"], label="test loss")
    plt.xticks([i*20 for i in range(10)])
    plt.yticks([i*0.2 for i in range(20)])
    plt.legend(loc="upper right")
    plt.grid()
    plt.sca(ax2)
    plt.plot(parser["val"]["epoch"], parser["val"]["acc"], label="test accurarcy")
    plt.xticks([i*20 for i in range(10)])
    plt.yticks([i*5 for i in range(20)])
    plt.legend(loc="lower right")
    plt.grid()

    plt.show()


