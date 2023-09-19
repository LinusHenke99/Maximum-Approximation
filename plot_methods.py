from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from plot_intervals import get_attributes, load_data


def filter() -> list[str]:
    dir_list = listdir("./data")
    dir_list = [
        element
        for element in dir_list
        if "gauss" in element and not "scaled" in element and "extended" in element
    ]

    attributes = get_attributes(dir_list, "gauss")

    dir_list = [
        element
        for element in dir_list
        if attributes[element]["mu"] == 0.0
        and attributes[element]["offset"] == 3.0
        and attributes[element]["polydeg"] == 5.0
        and attributes[element]["std"] == 1.0
    ]

    return dir_list


def main() -> None:
    files = filter()
    data = load_data(files)

    data = data[files[0]]

    Ns = np.array([int(key.split("=")[-1]) for key in data])

    plot_data = np.ndarray((3, len(Ns)))

    for second_index, N in enumerate(data):
        real_max = np.array(data[N]["real_max"])
        approx_max = np.array(data[N]["approx_max"])
        median_max = np.array(data[N]["median"])
        mean_max = np.array(data[N]["mean"])

        calc_mse = lambda x: np.sum((x - real_max) ** 2) / np.sum(real_max**2)

        plot_data[0, second_index] = calc_mse(approx_max)
        plot_data[1, second_index] = calc_mse(median_max)
        plot_data[2, second_index] = calc_mse(mean_max)

    plt.plot(Ns, plot_data[0], label="no batch utilization")
    plt.plot(Ns, plot_data[1], label="median of batch")
    plt.plot(Ns, plot_data[2], label="mean of batch")

    plt.legend()
    plt.grid()
    plt.xlabel("batch size")
    plt.savefig("./plots/methods.pdf", dpi=150)

    plt.show()


if __name__ == "__main__":
    main()
