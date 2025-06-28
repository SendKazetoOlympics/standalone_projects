from jaxtyping import Float, Int
from torch import Tensor


class Analyzer:
    data: list[tuple[Int, Int, Float[Tensor, "H W"]]]

    def __init__(self, data: list[tuple[Int, Int, Float[Tensor, "H W"]]]):
        self.data = data
