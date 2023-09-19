import numpy as np
from gauss_error_accumulation import calc_divide_and_conqer


def get_kernel_arrays(
    input: np.array, kernel_size: int, stride: int = 1
) -> list[np.array]:
    size = len(input)

    output = list()

    for i in range(len(input)):
        stride_offset = i * stride

        if stride_offset + kernel_size > size:
            break

        output.append(input[stride_offset : stride_offset + kernel_size])

    return output


def relu(input: np.array) -> np.array:
    return np.maximum(0, input)


def simulate_simple_nn() -> float:
    N1 = 28

    mu1, sig1 = 0.0, 1.0
    input_01 = np.random.normal(mu1, sig1, N1 * N1)

    kernels = get_kernel_arrays(input_01, 3 * 3)
    input_02_approx = np.array(
        [
            calc_divide_and_conqer(elem, 5, mu1 - 3 * sig1, mu1 + 3 * sig1)
            for elem in kernels
        ]
    )
    input_02_plain = np.array(
        [
            calc_divide_and_conqer(
                elem, 5, mu1 - 3 * sig1, mu1 + 3 * sig1, no_approx=True
            )
            for elem in kernels
        ]
    )

    N2 = 8
    mu2, sig2 = 0.0, 0.5
    matrix = np.random.normal(mu2, sig2, (N2, len(input_02_approx)))

    input_02_approx = matrix @ input_02_approx
    input_02_plain = matrix @ input_02_plain
    input_02_plain = relu(input_02_plain)
    input_02_approx = relu(input_02_approx)

    mu3, sig3 = np.mean(input_02_approx), np.std(input_02_approx)
    kernels_approx = get_kernel_arrays(input_02_approx, 2*2)
    kernels_plain = get_kernel_arrays(input_02_plain, 2*2)

    input_02_approx = np.array(
        [
            calc_divide_and_conqer(elem, 5, mu3 - 3 * sig3, mu3 + 3 * sig3)
            for elem in kernels_approx
        ]
    )
    input_02_plain = np.array(
        [
            calc_divide_and_conqer(
                elem, 5, mu1 - 3 * sig1, mu1 + 3 * sig1, no_approx=True
            )
            for elem in kernels_plain
        ]
    )

    print(input_02_approx)
    print(input_02_plain)


def main() -> None:
    simulate_simple_nn()


if __name__ == "__main__":
    main()
