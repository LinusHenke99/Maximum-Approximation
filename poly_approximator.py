from typing import Any
import numpy as np
from numpy.polynomial import Polynomial


class PolyApproximator:
    def get_polynomial(self) -> Polynomial:
        message = """Polynomial approximation has to be implemented by derived
        class."""
        raise NotImplementedError(message)

    def __init__(
        self, degree: int, min: float, max: float, n_points: int = 100
    ) -> None:
        self.min = min
        self.max = max
        self.degree = degree
        self.n_points = n_points

        self.x = np.linspace(self.min, self.max, self.n_points)
        self.y = np.abs(self.x)

        self.polynomial = self.get_polynomial()

    def update(self) -> None:
        self.x = np.linspace(self.min, self.max, self.n_points)
        self.y = np.abs(self.x)

        self.polynomial = self.get_polynomial()

    def __call__(self, x: np.array) -> np.array:
        return self.polynomial(x)


class ChebyshevApproximator(PolyApproximator):
    def __init__(
        self, degree: int, min: float, max: float, n_points: int = 100
    ) -> None:
        super().__init__(degree, min, max, n_points)

    def get_polynomial(self) -> Polynomial:
        coefs = (
            np.polynomial.chebyshev.Chebyshev.fit(self.x, self.y, self.degree)
            .convert()
            .coef
        )
        polynomial = Polynomial(coefs)

        return polynomial


class LaguereApproximator(PolyApproximator):
    def __init__(
        self, degree: int, min: float, max: float, n_points: int = 100
    ) -> None:
        super().__init__(degree, min, max, n_points)

    def get_polynomial(self) -> Polynomial:
        coefs = np.polynomial.Laguerre.fit(self.x, self.y, self.degree).convert().coef
        polynomial = Polynomial(coefs)

        return polynomial
