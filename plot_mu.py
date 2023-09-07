from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from plot_intervals import get_attributes, load_data


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

    return dir_list


def main() -> None:
    files = filter()

    data = load_data(files)

    mus = list(
        {get_attributes([filename], "gauss")[filename]["mu"] for filename in files}
    )
    mus.sort()
    mus = np.array(mus)

    plot_data = np.ndarray((5, len(mus)))

    for filename in data:
        attributes = get_attributes([filename], "gauss")[filename]

        mu = attributes["mu"]
        second_index = np.where(mus == mu)[0][0]

        for first_index, N in enumerate(data[filename]):
            real_max = np.array(data[filename][N]["real_max"])
            approx_max = np.array(data[filename][N]["approx_max"])

            mse = np.sum((real_max - approx_max) ** 2) / np.sum(real_max**2)

            plot_data[first_index, second_index] = mse

    N = [2, 4, 9, 16, 25]

    for dat, n in zip(plot_data, N):
        plt.scatter(mus, dat.flatten(), label=f"N={n}")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
