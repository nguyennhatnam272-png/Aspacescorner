#!/usr/bin/env python3
"""Turn the ERA5 monthly 2 m temperature climatology into the compact file the page loads.

Input:  source/era5-t2m-monthly.npz   (from fetch_era5_temperature.py)
        source/era5-climatology.npz   (from fetch_era5_pressure.py: ERA5 orography z)
Output: earth/earth-temperature.b64.txt  base64 of zlib-deflated data: 12 monthly grids of 720 x 361
        (0.5 deg, node-registered: lon -180..179.5, lat 90..-90), Int16 LE in 0.1 C, each row stored as differences
        from the previous value (the first value of a row as is), which compresses far better.

The grids hold the temperature reduced to sea level with a standard lapse rate, T_sl = T_2m + 0.0065 * z (ERA5's own
smoothed terrain). The page brings it back to the true ground height h from its DEM: T = T_sl - 0.0065 * h (h = 0 at sea),
so valleys and peaks finer than ERA5's grid get their own temperatures.
"""
import base64, pathlib, zlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
GAMMA = 0.0065


def main():
    t = np.load(HERE / 'source' / 'era5-t2m-monthly.npz')['t2m']          # 12 x 721 x 1440, C, lon 0..359.75
    z = np.load(HERE / 'source' / 'era5-climatology.npz')['z']            # 721 x 1440, m
    tsl = t + GAMMA * z[None]
    tsl = np.roll(tsl, 720, axis=2)[:, ::2, ::2]                           # lon -180..179.5, 0.5 deg: 12 x 361 x 720
    q = np.round(tsl * 10).astype(np.int32)
    d = q.copy(); d[:, :, 1:] = q[:, :, 1:] - q[:, :, :-1]
    assert d.min() >= -32768 and d.max() <= 32767
    out = HERE / 'earth' / 'earth-temperature.b64.txt'
    out.write_bytes(base64.b64encode(zlib.compress(d.astype('<i2').tobytes(), 9)))
    ann = t.mean(axis=0)
    w = np.cos(np.radians(np.linspace(90, -90, 721)))[:, None]
    print(out.name, round(out.stat().st_size / 1e6, 2), 'MB; global area-weighted mean', round(float((ann * w).sum() / (w.sum() * 1440)), 2), 'C',
          '; T_sl range', round(float(tsl.min()), 1), round(float(tsl.max()), 1))


if __name__ == '__main__':
    main()
