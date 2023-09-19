from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import json
from plot_intervals import get_attributes, load_data


def filter() -> list[str]:
    dir_list = listdir("./data")
    dir_list = [
        element
        for element in dir_list
        if "gauss" in element and not "scaled" in element and not "extended" in element
    ]

    attributes = get_attributes(dir_list, "gauss")

    dir_list = [
        element
        for element in dir_list
        if attributes[element]["mu"] == 0.0
        and attributes[element]["offset"] == 3.0
        and attributes[element]["polydeg"] == 5.0
    ]

    return dir_list


def main() -> None:
    files = filter()

    data = load_data(files)

    sigmas = list(
        {get_attributes([filename], "gauss")[filename]["std"] for filename in files}
    )
    sigmas.sort()
    sigmas = np.array(sigmas)

    plot_data = np.ndarray((5, len(sigmas)))

    for filename in data:
        attributes = get_attributes([filename], "gauss")[filename]

        sigma = attributes["std"]
        second_index = np.where(sigmas == sigma)[0][0]

        for first_index, N in enumerate(data[filename]):
            real_max = np.array(data[filename][N]["real_max"])
            approx_max = np.array(data[filename][N]["approx_max"])

            mse = np.sum((real_max - approx_max) ** 2) / np.sum(real_max**2)

            plot_data[first_index, second_index] = mse

    N = [2, 4, 9, 16, 25]

    for dat, n in zip(plot_data, N):
        plt.scatter(sigmas, dat.flatten(), label=f"N={n}")

    plt.legend()
    plt.grid()
    plt.title("Influence of the standard deviation")
    plt.xlabel("standard deviation")
    plt.savefig("./plots/standard_deviation.pdf", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
