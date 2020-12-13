from typing import Any, Dict, Tuple, List, Literal, Optional, Union
from typing import NamedTuple
import math
import numpy as np
import xarray as xr
from PIL import Image, ImageOps, ImageDraw


class Extent(NamedTuple):
    dim: str
    left: Optional[float]
    right: Optional[float]
    point_width: Optional[float]


def extent(da: xr.DataArray, dim: str, default: Optional[Extent] = None) -> Extent:
    if default is None:
        default = Extent(dim, None, None, None)
    coord = da[dim]
    n = len(coord.values)
    if n == 0:
        res = Extent(dim, None, None, default.point_width)
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
        defaults = [None for _ in da.dims]
    return [extent(da, k, defaults[i]) for i, k in enumerate(da.dims)]


def pad_to_extents(
    da: xr.DataArray,
    extents: List[Extent],
    default_point_widths: Optional[List[Optional[float]]] = None,
):
    pass


def g_lon(tx, tz):
    return tx * 360 / 2 ** tz - 180


def g_lat(tx, tz):
    return tx * 180 / 2 ** tz - 90


def g_lat_3857(tx, tz):
    a = 2 * math.pi * tx / 2 ** tz - math.pi
    return 180 / math.pi * math.atan(0.5 * (math.exp(a) - math.exp(-a)))


def tile_extents(g, tx, tz, n=1):
    assert n >= 1 and tz >= 0 and 0 <= tx < 2 ** tz
    a = g(tx, tz)
    for i in range(1, n + 1):
        b = g(tx + i / n, tz)
        yield a, b
        a = b


def produce_tile(ds, tx, ty, tz, tile_width=256, tile_height=256) -> Image:

    ds_es = extents(ds)
    print(ds_es, ds.dims)
    ex = list(tile_extents(g_lon, tx, tz, 10))
    ey = list(tile_extents(g_lat, tx, tz, 10))
    print(ex[0][0], ex[-1][-1])
    print(ey[0][0], ey[-1][-1])

    return None
    target_pixel_width = 1.0 / tile_width
    target_pixel_height = 1.0 / tile_height

    xs = np.fromiter((i / tile_width for i in range(tile_width + 1)), np.double)

    ys = np.fromiter((j / tile_height for j in range(tile_height + 1)), np.double)

    ts = np.fromiter((x / target_size[1] for x in range(target_size[1] + 1)), np.double)
    vs = np.fromiter((tile_to_y_3857(y + d, z) for d in ts), np.double)

    def _destination_rectangle(i):
        return (0, i, 256, i + 1)

    def _source_quadrilateral(i):
        return (
            0,
            vs[i],
            0,
            vs[i + 1],
            source_size[0] + 1,
            vs[i + 1],
            source_size[0] + 1,
            vs[i],
        )

    mesh = [
        (
            _destination_rectangle(i, j),
            _source_quadrilateral(i, j),
        )
        for i in range(256)
    ]
    im = image.transform(image.size, Image.MESH, mesh, Image.BILINEAR)
    return im


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

    bath_es = extents(bath)
    print(bath)
    print("*** bath es:", bath_es, bath.dims)

    da = bath.sel(X=slice(179, 179.1), Y=slice(88, 89.1)).transpose("Y", "X")
    print(da)

    es = extents(da, bath_es)
    print("*** es:", es, da.dims)

    values = da.values
    print(values.shape)
    values = np.pad(
        values, ((10, 0),), mode="constant", constant_values=((np.nan, np.nan),)
    )
    print(values.shape)
    print(values)

    # z = 0
    # print([(tile_to_lon(i, z), tile_to_lat(i, z), tile_to_lat_3857(i, z)) for i in range(0, 2 ** z)])

    im = produce_tile(bath, tx=1, ty=1, tz=2)
    # im.save(f"figs/fig-{z}-{x}-{y}.png", format="png")


main()
