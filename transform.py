import math
import numpy as np
import xarray as xr
from PIL import Image, ImageOps, ImageDraw


def extent(ds, coord_name):
    coord = ds[coord_name]
    point_width = coord.values[1] - coord.values[0]
    return (
        coord.values[0] - point_width / 2,
        coord.values[-1] + point_width / 2,
        point_width,
    )


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

    ds_ex = extent(ds, "X")
    ds_ey = extent(ds, "Y")
    print(ds_ex, ds_ey)
    es = tile_extents(g_lat_3857, tx, tz, 9)
    print(list(es))

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
    bath = xr.open_dataset("bath432.nc", decode_times=False)
    xs = np.fromiter(
        ((-180 + 0.41570438799076215) + i * 0.8314087759815243 for i in range(433)),
        np.double,
    )
    ys = np.fromiter(
        ((-90 + 0.4147465437788018) + i * 0.8294930875576036 for i in range(217)),
        np.double,
    )
    bath = bath.assign_coords(X=xs, Y=ys)
    lon_start, lon_stop, lon_point_width = extent(bath, "X")
    lat_start, lat_stop, lat_point_width = extent(bath, "Y")
    print(bath)
    print(lon_start, lon_stop, lon_point_width)
    print(lat_start, lat_stop, lat_point_width)

    # z = 0
    # print([(tile_to_lon(i, z), tile_to_lat(i, z), tile_to_lat_3857(i, z)) for i in range(0, 2 ** z)])

    im = produce_tile(bath, tx=1, ty=1, tz=2)
    # im.save(f"figs/fig-{z}-{x}-{y}.png", format="png")


main()
