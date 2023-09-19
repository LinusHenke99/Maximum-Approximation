import matplotlib.pyplot as plt
from os import listdir
from plot_intervals import load_data, get_attributes
import numpy as np


mus = [0.0, 0.5, 1.0, 1.25, 1.5]


def filter() -> list[str]:
    dir_list = listdir("./data")
    dir_list = [element for element in dir_list if "gauss" in element]

    attributes = get_attributes(dir_list, "gauss")

    dir_list = [
        element
        for element in dir_list
        if attributes[element]["std"] == 1.0
        and attributes[element]["offset"] == 3.0
        and attributes[element]["polydeg"] == 5.0
    ]

    attributes = get_attributes(dir_list, "gauss")

    dir_list = [
        element for element in dir_list if attributes[element]["mu"] in mus
    ]

    return dir_list


colors = ["green", "red", "blue", "yellow", "orange"]


def main() -> None:
    files = filter()

    data = load_data(files)

    sigmas = np.array(mus)

    plot_data = np.ndarray((5, len(sigmas)))
    plot_data_scaled = np.ndarray((5, len(sigmas)))

    for filename in data:
        lst = [filename]
        attributes = get_attributes(lst, "gauss")[filename]

        std = attributes["mu"]

        second_index = np.where(sigmas == std)[0][0]

        for first_index, N in enumerate(data[filename]):
            real_max = np.array(data[filename][N]["real_max"])
            approx_max = np.array(data[filename][N]["approx_max"])

            mse = np.sum((real_max - approx_max) ** 2) / np.sum(real_max**2)

            if "scaled" in filename:
                plot_data_scaled[first_index, second_index] = mse

            else:
                plot_data[first_index, second_index] = mse

    N = [2, 4, 9, 16, 25]

    for dat, n, color in zip(plot_data, N, colors):
        plt.plot(sigmas, dat.flatten(), label=f"N={n}", color=color)

    for dat, n, color in zip(plot_data_scaled, N, colors):
        plt.plot(sigmas, dat.flatten(), color=color, linestyle="dashed")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
