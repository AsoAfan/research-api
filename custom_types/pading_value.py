from typing import TypedDict


class Padding3D(TypedDict):
    pad_x: tuple[int, int]
    pad_y: tuple[int, int]
    pad_z: tuple[int, int]