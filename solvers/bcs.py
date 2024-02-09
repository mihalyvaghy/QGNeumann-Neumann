from enum import Enum

class BCType(Enum):
    DIRICHLET = 1
    NEUMANN = 2

class BC:
    def __init__(self, type, value=0):
        self.type = type
        self.value = value