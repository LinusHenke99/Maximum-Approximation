from os import listdir
import matplotlib.pyplot as plt
import json
import numpy as np


def get_attributes(filelist: list[str], distribution: str) -> dict:
    attributes = dict()
    dir_list = filelist.copy()

    for filename in dir_list:
        split = filename.split("_")
        split.remove(distribution)
        subdict = {
            element.split("=")[0]: float(element.split("=")[1].split("*")[0])
            for element in split
        }
        attributes[filename] = subdict

    return attributes


def filter() -> list[str]:
    dir_list = listdir("./data")
    dir_list = [element for element in dir_list if "gauss" in element]

    attributes = get_attributes(dir_list, "gauss")

    dir_list = [
        element
        for element in dir_list
        if attributes[element]["mu"] == 0.0
        and attributes[element]["std"] == 5.0
        and attributes[element]["polydeg"] == 5.0
    ]

    return dir_list


def main() -> None:
    files = filter()

    data = dict()
    offsets = np.linspace(1.0, 3.0, num=9)

    for filename in files:
        with open("./data/" + filename, "r") as jsonfile:
            json_object = json.load(jsonfile)

        data[filename] = json_object

    plot_data = np.ndarray((5, 9))
    plot_err = np.ndarray((5, 9))

    for filename in data:
        lst = [filename]
        attributes = get_attributes(lst, "gauss")[filename]

        offset = attributes["offset"]

        second_index = np.where(offsets == offset)[0][0]

        for first_index, N in enumerate(data[filename]):
            real_max = np.array(data[filename][N]["real_max"])
            approx_max = np.array(data[filename][N]["approx_max"])

            # overflow = np.where(np.abs(approx_max) >= 1e20)
            # print(overflow)

            # if len(overflow[0]) > 0:
            #     for index in overflow[0]:
            #         print(index)
            #         real_max[index] = 0
            #         approx_max[index] = 0
            # print(filename, N)
            # print(f"Shape of real_max={real_max.shape}, approx={approx_max.shape}")

            sq_err = (real_max - approx_max) ** 2

            msq_err = np.mean(sq_err)
            std_dev = np.std(sq_err)

            plot_data[first_index, second_index] = msq_err
            plot_err[first_index, second_index] = std_dev

    N = [2, 4, 9, 16, 25]

    for err, dat, n in zip(plot_err, plot_data, N):
        plt.scatter(offsets, dat.flatten(), label=f"N={n}")

    plt.loglog()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
