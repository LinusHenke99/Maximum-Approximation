import json
import numpy as np
from gauss_error_accumulation import calc_maxs
from tqdm import tqdm
import matplotlib.pyplot as plt


TRIES = 100
MU = 0.0
STD = 1.0
std = [1.0, 0.5, 1.25, 1.5]
POLY_DEG = 5
FACTOR = 3.0
NS = [2, 4, 9, 16, 25]

TARGET_MU = 1.25
TARGET_SIGMA = 1.0


def first_trafo(inp: np.array, mu: float, sigma: float) -> np.array:
    result = inp - mu + TARGET_MU

    return result


def second_trafo(inp: np.array, mu: float, sigma: float) -> np.array:
    result = inp - TARGET_MU
    result += mu

    return result


SCALER = (first_trafo, second_trafo)


def main() -> None:
    experiment = dict()

    for mu in tqdm(std):
        for N in NS:
            print(f"For N={N}")
            data = {
                "real_max": list(),
                "approx_max": list(),
                "median": list(),
                "mean": list(),
            }

            for _ in tqdm(range(TRIES)):
                real_max, approx_max, median, mean = calc_maxs(
                    N, MU, STD, POLY_DEG, FACTOR, scaler=SCALER
                )
                values = [real_max, approx_max, median, mean]

                for key, value in zip(data, values):
                    data[key].append(value)

            experiment[f"N={N}"] = data

        json_object = json.dumps(experiment, indent=4)

        filename = f"gauss_scaled_mu={MU}_std={STD}_polydeg={POLY_DEG}_offset={FACTOR}*std.json"
        with open("./data/" + filename, "w") as outfile:
            outfile.write(json_object)


if __name__ == "__main__":
    main()
