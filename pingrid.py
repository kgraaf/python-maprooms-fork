from typing import Any, Dict, Tuple, List, Literal, Optional, Union, Callable, Hashable
from typing import NamedTuple
import math
import datetime
import numpy as np
import pandas as pd
import xarray as xr
from collections.abc import Iterable
from scipy import interpolate
import cv2
import psycopg2.extensions
from queuepool.psycopg2cm import ConnectionManagerExtended
from queuepool.pool import Pool
import rasterio.features
import rasterio.transform
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipoint import MultiPoint


def init_dbpool(name, config):
    dbpoolConf = config[name]
    dbpool = Pool(
        name=dbpoolConf["name"],
        capacity=dbpoolConf["capacity"],
        maxIdleTime=dbpoolConf["max_idle_time"],
        maxOpenTime=dbpoolConf["max_open_time"],
        maxUsageCount=dbpoolConf["max_usage_count"],
        closeOnException=dbpoolConf["close_on_exception"],
    )
    for i in range(dbpool.capacity):
        dbpool.put(
            ConnectionManagerExtended(
                name=dbpool.name + "-" + str(i),
                autocommit=False,
                isolation_level=psycopg2.extensions.ISOLATION_LEVEL_SERIALIZABLE,
                dbname=dbpool.name,
                host=dbpoolConf["host"],
                port=dbpoolConf["port"],
                user=dbpoolConf["user"],
                password=dbpoolConf["password"],
            )
        )
    if dbpoolConf["recycle_interval"] is not None:
        dbpool.startRecycler(dbpoolConf["recycle_interval"])
    return dbpool


FuncInterp2d = Callable[[np.ndarray, np.ndarray], np.ndarray]


class DataArrayEntry(NamedTuple):
    name: str
    data_array: xr.DataArray
    interp2d: Dict[Any, FuncInterp2d]
    min_val: Optional[float]
    max_val: Optional[float]
    colormap: Optional[np.ndarray]


class Extent(NamedTuple):
    dim: str
    left: float
    right: float
    point_width: float


class BGR(NamedTuple):
    blue: int
    green: int
    red: int


class BGRA(NamedTuple):
    blue: int
    green: int
    red: int
    alpha: int


class DrawAttrs(NamedTuple):
    line_color: Union[int, BGR, BGRA]
    background_color: Union[int, BGR, BGRA]
    line_thickness: int
    line_type: int  # cv2.LINE_4 | cv2.LINE_8 | cv2.LINE_AA


def create_interp2d(
    da: xr.DataArray, dims: Tuple[str, str]
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    x = da[dims[1]].values
    y = da[dims[0]].values
    z = da.values
    f = interpolate.interp2d(x, y, z, kind="linear", copy=False, bounds_error=False)
    return f


def from_months_since(x, year_since=1960):
    int_x = int(x)
    return datetime.date(
        int_x // 12 + year_since, int_x % 12 + 1, int(30 * (x - int_x) + 1)
    )


from_months_since_v = np.vectorize(from_months_since)


def to_months_since(d, year_since=1960):
    return (d.year - year_since) * 12 + (d.month - 1) + (d.day - 1) / 30.0


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


def extents(
    da: xr.DataArray, dims: List[str] = None, defaults: Optional[List[Extent]] = None
) -> List[Extent]:
    if dims is None:
        dims = da.dims
    if defaults is None:
        defaults = [Extent(dim, np.nan, np.nan, np.nan) for dim in dims]
    return [extent(da, k, defaults[i]) for i, k in enumerate(dims)]


def g_lon(tx: int, tz: int) -> float:
    return tx * 360 / 2 ** tz - 180


def g_lat(tx: int, tz: int) -> float:
    return tx * 180 / 2 ** tz - 90


def g_lat_3857(tx: int, tz: int) -> float:
    a = math.pi - 2 * math.pi * tx / 2 ** tz
    return 180 / math.pi * math.atan(0.5 * (math.exp(a) - math.exp(-a)))


def tile_extents(g: Callable[[int, int], float], tx: int, tz: int, n: int = 1):
    assert n >= 1 and tz >= 0 and 0 <= tx < 2 ** tz
    a = g(tx, tz)
    for i in range(1, n + 1):
        b = g(tx + i / n, tz)
        yield a, b
        a = b


def produce_data_tile(
    interp2d: Callable[[np.ndarray, np.ndarray], np.ndarray],
    tx: int,
    ty: int,
    tz: int,
    tile_width: int = 256,
    tile_height: int = 256,
) -> np.ndarray:
    x = np.fromiter(
        (a + (b - a) / 2.0 for a, b in tile_extents(g_lon, tx, tz, tile_width)),
        np.double,
    )
    y = np.fromiter(
        (a + (b - a) / 2.0 for a, b in tile_extents(g_lat_3857, ty, tz, tile_height)),
        np.double,
    )
    # print("*** produce_data_tile:", tz, tx, ty, x[0], x[-1], y[0], y[-1])
    z = interp2d(x, y)
    return z


def to_multipolygon(p: Union[Polygon, MultiPolygon]) -> MultiPolygon:
    if not isinstance(p, MultiPolygon):
        p = MultiPolygon([p])
    return p


def produce_shape_tile(
    im: np.ndarray,
    shapes: List[Tuple[MultiPolygon, DrawAttrs]],
    tx: int,
    ty: int,
    tz: int,
    tile_width: int = 256,
    tile_height: int = 256,
) -> np.ndarray:
    x0, x1 = list(tile_extents(g_lon, tx, tz, 1))[0]
    y0, y1 = list(tile_extents(g_lat_3857, ty, tz, 1))[0]

    tile_bounds = (x0, y0, x1, y1)
    tile = MultiPoint([(x0, y0), (x1, y1)]).envelope

    for s, a in shapes:
        m = to_multipolygon(s.intersection(tile))
        for p in m:
            if not p.is_empty:
                xs, ys = p.exterior.coords.xy
                xs = ((np.array(xs) - x0) / (x1 - x0) * tile_width).astype(np.int32)
                ys = ((np.array(ys) - y0) / (y1 - y0) * tile_height).astype(
                    np.int32
                )  # TODO: apply mercator transform
                pts = np.column_stack((xs, ys))
                pts = pts.reshape((1,) + pts.shape)
                print("*** pts:", pts, pts.shape, tx, ty, tz, a.background_color)
                cv2.fillPoly(im, pts, a.background_color)

    return im


def produce_test_tile(w: int = 256, h: int = 256, text: str = "") -> np.ndarray:
    line_thickness = 1
    line_type = cv2.LINE_AA  # cv2.LINE_4 | cv2.LINE_8 | cv2.LINE_AA
    red_color = BGRA(0, 0, 255, 255)
    green_color = BGRA(0, 255, 0, 255)
    blue_color = BGRA(255, 0, 0, 255)

    layer1 = np.zeros((h, w, 4), np.uint8)
    layer2 = np.zeros((h, w, 4), np.uint8)
    layer3 = np.zeros((h, w, 4), np.uint8)

    cv2.ellipse(
        layer1,
        (w // 2, h // 2),
        (w // 3, h // 3),
        0,
        0,
        360,
        red_color,
        line_thickness,
        lineType=line_type,
    )

    cv2.putText(
        layer1,
        text,
        (w // 2, h // 2),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=red_color,
        thickness=1,
        lineType=line_type,
    )

    cv2.rectangle(
        layer2, (0, 0), (w - 1, h - 1), green_color, line_thickness, lineType=line_type
    )
    cv2.line(
        layer3, (0, 0), (w - 1, h - 1), blue_color, line_thickness, lineType=line_type
    )
    cv2.line(
        layer3, (0, h - 1), (w - 1, 0), blue_color, line_thickness, lineType=line_type
    )

    res = layer1[:]
    cnd = layer2[:, :, 3] > 0
    res[cnd] = layer2[cnd]
    cnd = layer3[:, :, 3] > 0
    res[cnd] = layer3[cnd]

    return res


def correct_coord(da: xr.DataArray, coord_name: str) -> xr.DataArray:
    coord = da[coord_name]
    values = coord.values
    min_val = values[0]
    max_val = values[-1]
    n = len(values)
    point_width = (max_val - min_val) / n
    vs = np.fromiter(
        ((min_val + point_width / 2.0) + i * point_width for i in range(n)), np.double
    )
    return da.assign_coords({coord_name: vs})


def parse_color(s: str) -> BGRA:
    v = int(s)
    return BGRA(v >> 0 & 0xFF, v >> 8 & 0xFF, v >> 16 & 0xFF, 255)


def parse_color_item(vs: List[BGRA], s: str) -> List[BGRA]:
    if s == "null":
        rs = [BGRA(0, 0, 0, 0)]
    elif s[0] == "[":
        rs = [parse_color(s[1:])]
    elif s[-1] == "]":
        n = int(s[:-1])
        assert 1 < n <= 256 and len(vs) != 0
        rs = [vs[-1]] * (n - 1)
    else:
        rs = [parse_color(s)]
    return vs + rs


def parse_colormap(s: str) -> np.ndarray:
    vs = []
    for x in s[1:-1].split(" "):
        vs = parse_color_item(vs, x)
    # print("*** CM cm:", len(vs), [f"{v.red:02x}{v.green:02x}{v.blue:02x}" for v in vs])
    rs = np.array([vs[int(i / 256.0 * len(vs))] for i in range(0, 256)], np.uint8)
    return rs


def apply_colormap(im: np.ndarray, colormap: np.ndarray) -> np.ndarray:
    im = im.astype(np.uint8)
    im = cv2.merge(
        [
            cv2.LUT(im, colormap[:, 2]),
            cv2.LUT(im, colormap[:, 1]),
            cv2.LUT(im, colormap[:, 0]),
            cv2.LUT(im, colormap[:, 3]),
        ]
    )
    return im


#
# Functions to deal with spatial averaging
#


def trim_to_bbox(ds, s, lon_name="lon", lat_name="lat"):
    """Given a Dataset and a shape, return the subset of the Dataset that
    intersects the shape's bounding box.
    """
    lon_res = ds[lon_name].values[1] - ds[lon_name].values[0]
    lat_res = ds[lat_name].values[1] - ds[lat_name].values[0]

    lon_min, lat_min, lon_max, lat_max = s.bounds
    # print("*** shape bounds:", lon_min, lat_min, lon_max, lat_max, file=sys.stderr)

    lon_min -= 1 * lon_res
    lon_max += 1 * lon_res
    lat_min -= 1 * lat_res
    lat_max += 1 * lat_res

    return ds.sel(
        {lon_name: slice(lon_min, lon_max), lat_name: slice(lat_min, lat_max)}
    )


def average_over(ds, s, lon_name="lon", lat_name="lat", all_touched=False):
    """Average a Dataset over a shape"""
    lon_res = ds[lon_name].values[1] - ds[lon_name].values[0]
    lat_res = ds[lat_name].values[1] - ds[lat_name].values[0]

    lon_min = ds[lon_name].values[0] - 0.5 * lon_res
    lon_max = ds[lon_name].values[-1] + 0.5 * lon_res
    lat_min = ds[lat_name].values[0] - 0.5 * lat_res
    lat_max = ds[lat_name].values[-1] + 0.5 * lat_res

    lon_size = ds.sizes[lon_name]
    lat_size = ds.sizes[lat_name]

    t = rasterio.transform.Affine(
        (lon_max - lon_min) / lon_size,
        0,
        lon_min,
        0,
        (lat_max - lat_min) / lat_size,
        lat_min,
    )

    r0 = rasterio.features.rasterize(
        s, out_shape=(lat_size, lon_size), transform=t, all_touched=all_touched
    )
    r0 = xr.DataArray(
        r0,
        dims=(lat_name, lon_name),
        coords={lat_name: ds[lat_name], lon_name: ds[lon_name]},
    )
    r = r0 * np.cos(np.deg2rad(ds[lat_name]))

    res = (ds * r).sum([lat_name, lon_name], skipna=True)
    res = res / (~np.isnan(ds) * r).sum([lat_name, lon_name])

    res.name = ds.name
    return res


def average_over_trimmed(ds, s, lon_name="lon", lat_name="lat", all_touched=False):
    ds = trim_to_bbox(ds, s, lon_name=lon_name, lat_name=lat_name)
    res = average_over(
        ds, s, lon_name=lon_name, lat_name=lat_name, all_touched=all_touched
    )
    return res


#
# Functions to deal with periodic dimension (e.g. longitude)
#


def __dim_range(ds, dim, period=360.0):
    c0, c1 = ds[dim].values[0], ds[dim].values[-1]
    d = (period - (c1 - c0)) / 2.0
    c0, c1 = c0 - d, c1 + d
    return c0, c1


def __normalize_vals(v0, vals, period=360.0, right=False):

    vs = vals if isinstance(vals, Iterable) else [vals]

    v1 = v0 + period
    assert v0 <= 0.0 <= v1

    vs = np.mod(vs, period)
    if right:
        vs[vs > v1] -= period
    else:
        vs[vs >= v1] -= period

    vs = vs if isinstance(vals, Iterable) else vs[0]

    return vs


def __normalize_dim(ds, dim, period=360.0):
    """Doesn't copy ds. Make a copy if necessary."""
    c0, c1 = __dim_range(ds, dim, period)
    if c0 > 0.0:
        ds[dim] = ds[dim] - period
    elif c1 < 0.0:
        ds[dim] = ds[dim] + period


def roll_to(ds, dim, val, period=360.0):
    """Rolls the ds to the first dim's label that is greater or equal to
    val, and then makes dim monitonically increasing. Assumes that dim
    is monotonically increasing, covers exactly one period, and overlaps
    val. If val is outside of the dim, this function does nothing.
    """
    a = np.argwhere(ds[dim].values >= val)
    n = a[0, 0] if a.shape[0] != 0 else 0
    if n != 0:
        ds = ds.copy()
        ds = ds.roll(**{dim: -n}, roll_coords=True)
        ds[dim] = xr.where(ds[dim] < val, ds[dim] + period, ds[dim])
        __normalize_dim(ds, dim, period)
    return ds


def sel_periodic(ds, dim, vals, period=360.0):
    """Assumes that dim is monotonically increasing, covers exactly one period, and overlaps 0.0
    Examples: lon: 0..360, -180..180, -90..270, -360..0, etc.
    TODO: change API to match xarray's `sel`
    """
    c0, c1 = __dim_range(ds, dim, period)
    print(f"*** sel_periodic (input): {c0}..{c1}: {vals}")

    if isinstance(vals, slice):
        if vals.step is None or vals.step >= 0:
            s0 = __normalize_vals(c0, vals.start, period)
            s1 = __normalize_vals(c0, vals.stop, period, True)
        else:
            s0 = __normalize_vals(c0, vals.stop, period)
            s1 = __normalize_vals(c0, vals.start, period, True)

        print(f"*** sel_periodic (normalized): {c0}..{c1}: {s0=}, {s1=}")

        if s0 > s1:
            ds = roll_to(ds, dim, s1, period)
            c0, c1 = __dim_range(ds, dim, period)
            s0 = __normalize_vals(c0, s0, period)
            s1 = __normalize_vals(c0, s1, period, True)
            print(f"*** sel_periodic (rolled): {c0}..{c1}: {s0=}, {s1=}")

        if vals.step is None or vals.step >= 0:
            vals = slice(s0, s1, vals.step)
        else:
            vals = slice(s1, s0, vals.step)

        print(f"*** sel_periodic (slice): {c0}..{c1}: {vals}")

    else:
        vals = __normalize_vals(c0, vals, period=period)
        print(f"*** sel_periodic (array): {c0}..{c1}: {vals}")

    ds = ds.sel({dim: vals})

    return ds
