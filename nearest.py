from typing import Tuple, List, Iterable, Callable
import numpy as np


def nearest_interpolator(
    input_grids: Iterable[Tuple[float, float]],  # [(x0, dx), (y0, dy), ...]
    input_data: np.ndarray,
) -> Callable[Iterable[np.ndarray], np.ndarray]:
    padded_data = np.pad(
        input_data, pad_width=1, mode="constant", constant_values=np.nan
    )

    def interp_func(output_grids: Iterable[np.ndarray]) -> np.ndarray:
        index = tuple(
            np.minimum(np.maximum(((x - (x0 - 1.5 * dx)) / dx).astype(int), 0), n - 1)
            for (x0, dx), x, n in zip(input_grids, output_grids, padded_data.shape)
        )
        return padded_data[tuple(reversed(np.meshgrid(*reversed(index))))]

    return interp_func


def nearest_all_together(
    input_grids: List[Tuple[float, float]],  # [(x0, dx), (y0, dy), ...]
    input_data: np.ndarray,
    output_grids: List[np.ndarray],  # [x, y, ...]
) -> np.ndarray:
    padded_data = np.pad(
        input_data, pad_width=1, mode="constant", constant_values=np.nan
    )
    index = tuple(
        np.minimum(np.maximum(((x - (x0 - 1.5 * dx)) / dx).astype(int), 0), n - 1)
        for (x0, dx), x, n in zip(input_grids, output_grids, padded_data.shape)
    )
    return padded_data[tuple(reversed(np.meshgrid(*reversed(index))))]


input_grids = [(1, 0.25), (2, 0.5)]
input_data = np.array([[10.0, 20.0, 30], [40.0, 50.0, 60.0]])
output_grids = [np.asarray([1.20, 1.25, 1.38]), np.asarray([1.95, 2.05, 2.1, 3.1])]

# input_grids = [(1, 0.25), (2, 0.5)]
# input_data = np.array([[10.0, 20.0, 30], [40.0, 50.0, 60.0]])
# output_grids = [np.asarray([1, 1.25]), np.asarray([2, 2.5, 3])]

print("*** input_grids:", input_grids)
print("*** output_grids:", output_grids)
print("*** input_data:", input_data, input_data.shape)
z = nearest_all_together(input_grids, input_data, output_grids)
print("*** z:", z)

f = nearest_interpolator(input_grids, input_data)
z2 = f(output_grids)
print("*** z1:", z2)

