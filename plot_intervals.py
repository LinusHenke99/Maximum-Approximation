from os import listdir
import matplotlib.pyplot as plt
import json
import numpy as np


ylabel = "normalized mean squared error"
plt.ylabel(ylabel)


def get_attributes(filelist: list[str], distribution: str) -> dict:
    attributes = dict()
    dir_list = filelist.copy()

    for filename in dir_list:
        scaled = False
        extended = False

        if "scaled" in filename:
            scaled = True

        if "extended" in filename:
            extended = True

        split = filename.split("_")
        split.remove(distribution)

        if scaled:
            split.remove("scaled")

        if extended:
            split.remove("extended")

        subdict = {
            element.split("=")[0]: float(element.split("=")[1].split("*")[0])
            for element in split
        }
        attributes[filename] = subdict

    return attributes


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
        and attributes[element]["std"] == 5.0
        and attributes[element]["polydeg"] == 5.0
    ]

    return dir_list


def load_data(files: list[str]) -> dict:
    data = dict()

    for filename in files:
        with open("./data/" + filename, "r") as jsonfile:
            json_object = json.load(jsonfile)

        data[filename] = json_object

    return data


def main() -> None:
    files = filter()

    offsets = np.linspace(1.0, 3.0, num=9)

    data = load_data(files)

    plot_data = np.ndarray((5, 9))

    for filename in data:
        lst = [filename]
        attributes = get_attributes(lst, "gauss")[filename]

        offset = attributes["offset"]

        second_index = np.where(offsets == offset)[0][0]

        for first_index, N in enumerate(data[filename]):
            real_max = np.array(data[filename][N]["real_max"])
            approx_max = np.array(data[filename][N]["approx_max"])

            msq_err = np.sum((real_max - approx_max) ** 2) / np.sum(real_max**2)

            plot_data[first_index, second_index] = msq_err

    N = [2, 4, 9, 16, 25]

    for dat, n in zip(plot_data, N):
        plt.plot(offsets[1:], dat[1:].flatten(), label=f"N={n}")

    plt.loglog()
    plt.legend()
    plt.grid()
    plt.xlabel("$\\lambda$")
    plt.show()


if __name__ == "__main__":
    main()
