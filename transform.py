from typing import Any, Dict, Tuple, List, Literal, Optional, Union, Callable, Hashable
from typing import NamedTuple
import math
import numpy as np
import xarray as xr
from scipy import interpolate
import cv2


class Extent(NamedTuple):
    dim: str
    left: float
    right: float
    point_width: float


def extent(da: xr.DataArray, dim: str, default: Optional[Extent] = None) -> Extent:
    if default is None:
        default = Extent(dim, np.nan, np.nan, np.nan)
    coord = da[dim]
    n = len(coord.values)
    if n == 0:
        res = Extent(dim, np.nan, np.nan, default.point_width)
    elif n == 1:
        res = Extent(
            dim,
            coord.values[0] - default.point_width / 2,
            coord.values[0] + default.point_width / 2,
            default.point_width,
        )
    else:
        point_width = coord.values[1] - coord.values[0]
        res = Extent(
            dim,
            coord.values[0] - point_width / 2,
            coord.values[-1] + point_width / 2,
            point_width,
        )
    return res


def extents(da: xr.DataArray, defaults: Optional[List[Extent]] = None) -> List[Extent]:
    if defaults is None:
        defaults = [Extent(dim, np.nan, np.nan, np.nan) for dim in da.dims]
    return [extent(da, k, defaults[i]) for i, k in enumerate(da.dims)]


def g_lon(tx: int, tz: int) -> float:
    return tx * 360 / 2 ** tz - 180


def g_lat(tx: int, tz: int) -> float:
    return tx * 180 / 2 ** tz - 90


def g_lat_3857(tx: int, tz: int) -> float:
    a = 2 * math.pi * tx / 2 ** tz - math.pi
    return 180 / math.pi * math.atan(0.5 * (math.exp(a) - math.exp(-a)))


def tile_extents(g: Callable[[int, int], float], tx: int, tz: int, n: int = 1):
    assert n >= 1 and tz >= 0 and 0 <= tx < 2 ** tz
    a = g(tx, tz)
    for i in range(1, n + 1):
        b = g(tx + i / n, tz)
        yield a, b
        a = b


def create_interp2d(
    da: xr.DataArray, dims: Tuple[str, str]
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    x = da[dims[1]].values
    y = da[dims[0]].values
    z = da.values
    f = interpolate.interp2d(x, y, z, kind="linear", copy=False, bounds_error=True)
    return f

def produce_tile(
    interp2d: Callable[[np.ndarray, np.ndarray], np.ndarray],
    tx: int,
    ty: int,
    tz: int,
    tile_width: int = 256,
    tile_height: int = 256,
) -> np.ndarray:
    x = np.fromiter((a + (b - a) / 2.0 for a, b in tile_extents(g_lon, tx, tz, tile_width)), np.double)
    y = np.fromiter((a + (b - a) / 2.0 for a, b in tile_extents(g_lat_3857, tx, tz, tile_height)), np.double)
    z = interp2d(x, y)
    return z


def main():
    bath = xr.open_dataset("bath432.nc", decode_times=False)["bath"].transpose("Y", "X")
    xs = np.fromiter(
        ((-180 + 0.41570438799076215) + i * 0.8314087759815243 for i in range(433)),
        np.double,
    )
    ys = np.fromiter(
        ((-90 + 0.4147465437788018) + i * 0.8294930875576036 for i in range(217)),
        np.double,
    )
    bath = bath.assign_coords(X=xs, Y=ys)
    print("*** bath:", bath)
    print("*** bath.dims:", bath.dims)
    interp2d = create_interp2d(bath, bath.dims)
    z = produce_tile(interp2d, tx=0, ty=0, tz=0, tile_width=256, tile_height=256)
    print(z)

main()
