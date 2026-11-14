#!/usr/bin/env python3
"""Download a monthly 2 m air temperature climatology (1991-2020) from ERA5.

Source: ECMWF ERA5 reanalysis, copy hosted by NSF NCAR on AWS Open Data (s3://nsf-ncar-era5, dataset ds633.0).
Licence: Copernicus licence; credit "Contains modified Copernicus Climate Change Service information (1991-2020)".

As in fetch_era5_pressure.py, whole monthly files (~1 GB) are not downloaded: one 27-hour compressed block is read over
HTTP byte ranges and averaged (which also averages out the day-night cycle). For each calendar month, 10 years spread
over 1991-2020 are sampled, 3 blocks per month (near days 1, 12 and 23): 30 sampled days per month, 360 in all.

Output: source/era5-t2m-monthly.npz with t2m (12 x 721 x 1440, degrees C, float32; lat 90..-90, lon 0..359.75).
Needs: numpy, h5py, fsspec, aiohttp.   Run:  python3 fetch_era5_temperature.py   (~4 min)
"""
import concurrent.futures as cf, pathlib, time
import numpy as np, fsspec, h5py

BASE = 'https://nsf-ncar-era5.s3.amazonaws.com/'
OUT = pathlib.Path(__file__).resolve().parent / 'source' / 'era5-t2m-monthly.npz'
FS = fsspec.filesystem('http', block_size=2 ** 22)


def month_file(y, m):
    nd = [31, 29 if y % 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
    return f'{BASE}e5.oper.an.sfc/{y}{m:02d}/e5.oper.an.sfc.128_167_2t.ll025sc.{y}{m:02d}0100_{y}{m:02d}{nd}23.nc'


def block_mean(job):
    url, blk = job
    err = None
    for attempt in range(6):
        try:
            with FS.open(url, 'rb', cache_type='blockcache') as f:
                d = h5py.File(f, 'r')['VAR_2T']
                c = d.chunks[0]
                return d[blk * c:(blk + 1) * c].astype(np.float64).mean(axis=0)
        except Exception as e:  # network hiccup: retry
            err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f'{url}: {err}')


def main():
    out = np.zeros((12, 721, 1440), np.float32)
    t0 = time.time()
    with cf.ThreadPoolExecutor(8) as ex:
        for m in range(1, 13):
            years = [1991 + 3 * i + (m % 3) for i in range(10)]
            jobs = [(month_file(y, m), b) for y in years for b in (0, 10, 20)]
            acc = None
            for a in ex.map(block_mean, jobs):
                acc = a if acc is None else acc + a
            out[m - 1] = (acc / len(jobs) - 273.15).astype(np.float32)
            print(f'month {m}: {len(jobs)} samples, global mean {out[m - 1].mean():.2f} C, {time.time() - t0:.0f}s', flush=True)
    np.savez_compressed(OUT, t2m=out, samples_per_month=30)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
