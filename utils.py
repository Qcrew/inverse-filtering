from enum import Enum


class PulseType(Enum):
    SQUARE_HOLD = 0
    SQUARE_RESET = 1
    GAUSSIAN = 2
    CONSTANT_COSINE = 3
    CONSTANT_COSINE_RESET = 4
    CUSTOM = 5


class DataType(Enum):
    PI_SCOPE = 0
    LINE_RESPONSE = 1


class SweepType(Enum):
    RANGE = 0
    VALUES = 1
