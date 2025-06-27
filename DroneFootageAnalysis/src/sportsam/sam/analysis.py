from jaxtyping import Float, Int, Array

class Analyzer:
    data: list[tuple[Int, Int, Float[Array, "H W"]]]
    
    def __init__(self, data: list[tuple[Int, Int, Float[Array, "H W"]]]):
        self.data = data
