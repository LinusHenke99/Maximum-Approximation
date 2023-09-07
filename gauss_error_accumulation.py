from poly_approximator import ChebyshevApproximator
import numpy as np
from tqdm import tqdm
import json

CALLED = 0


def get_pairwise_maximum(
    input: np.array, poly_deg: int, min: float, max: float, no_approx=False
) -> float:
    if not input.shape == (2,):
        raise ValueError(
            "Could not compute pairwise maximum, input array must have shape (2,)"
        )

    if no_approx:
        if input[0] > input[1]:
            return input[0]
        else:
            return input[1]

    approximator = ChebyshevApproximator(poly_deg, min, max)

    a, b = input
    result = 0.5 * (a + b + approximator(a - b))

    return result


def calc_max_brute_force(
    input: np.array, poly_deg: int, min: float, max: float
) -> np.array:
    first = input[0]
    input_copy = input[1:]

    for elem in input_copy:
        arr = np.array([first, elem])

        first = get_pairwise_maximum(arr, poly_deg, min, max)

    return first


def calc_divide_and_conqer(
    input: np.array, poly_deg: int, min: float, max: float, no_approx=False
) -> np.array:
    if input.shape == (2,):
        return get_pairwise_maximum(input, poly_deg, min, max, no_approx=no_approx)
    elif input.shape == (1,):
        return input[0]

    l, r = np.array_split(input, 2)

    max_l = calc_divide_and_conqer(l, poly_deg, min, max, no_approx=no_approx)
    max_r = calc_divide_and_conqer(r, poly_deg, min, max, no_approx=no_approx)

    arr = np.array([max_l, max_r])

    return get_pairwise_maximum(arr, poly_deg, min, max, no_approx=no_approx)


TRIES = 100
MU = 0.0
std = [9]
POLY_DEG = 5
FACTOR = 3.0
NS = [2, 4, 9, 16, 25]


def calc_maxs(N, MU, STD, POLY_DEG, FACTOR) -> tuple[float, float, float]:
    arr = np.zeros(N)
    arr = np.random.normal(MU, STD, N)

    real_max = np.max(arr)
    rolled = list()
    for i in range(N):
        roll = np.roll(arr, i)
        rolled.append(roll)

    approx_max = [
        calc_divide_and_conqer(inp, POLY_DEG, MU - FACTOR * STD, MU + FACTOR * STD)
        for inp in rolled
    ]

    median = np.median(approx_max)
    mean = np.mean(approx_max)

    return real_max, approx_max[0], median, mean


def squared_error(a: np.array, b: np.array) -> np.array:
    if a.shape != b.shape:
        raise ValueError("Arrays must have same size.")

    return (a - b) ** 2


if __name__ == "__main__":
    experiment = dict()

    for STD in tqdm(std):
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
                    N, MU, STD, POLY_DEG, FACTOR
                )
                values = [real_max, approx_max, median, mean]

                for key, value in zip(data, values):
                    data[key].append(value)

            experiment[f"N={N}"] = data

        json_object = json.dumps(experiment, indent=4)

        filename = (
            f"gauss_mu={MU}_std={STD}_polydeg={POLY_DEG}_offset={FACTOR}*std.json"
        )
        with open("./data/" + filename, "w") as outfile:
            outfile.write(json_object)
