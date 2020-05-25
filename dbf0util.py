import base64
import operator
import os
import pickle
import struct
from functools import reduce, partial
from io import FileIO
from typing import Callable, Union, Iterable, Optional, List, cast

import csaps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.axes import Axes
from tdigest import TDigest
from tqdm import tqdm

background_color = '#fcfcfc'
colors = ['#008fd5', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b', '#810f7c', '#cc6600', '#cc3399', '#00cc66']
number_t = Union[int, float]


def configure_notebook():
    plt.style.use(['fivethirtyeight'])
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 400)
    pd.set_option('display.max_colwidth', 100)

    tqdm.pandas()

    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

    old_background_color = plt.rcParams['axes.edgecolor']
    for k, v in plt.rcParams.items():
        if v == old_background_color:
            plt.rcParams[k] = background_color


def compose(*functions: Callable):
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def format_number(n: number_t, wrap=True):
    if isinstance(n, int) and n < 1000:
        ns = str(n)
    elif 1e-1 < n < 1e2:
        ns = f'{n:.2f}'
    else:
        p = int(np.floor(np.log10(n)))
        base = n / 10 ** p
        ns = f'{base:.2f}\\times 10^{{{p}}}'
    return f'${ns}$' if wrap else ns


def add_bar_values(ax: Axes, x: Iterable, offset: int = 2):
    for i, v in enumerate(x):
        ax.text(v + offset, i, f'{v:.1f}',
                verticalalignment='center',
                bbox=dict(facecolor=background_color, alpha=1))


def compute_rate(df: pd.DataFrame, y_label: str, dy_dt_label: Optional[str] = None,
                 result_attr: str = 'results', time_attr: str = 'time',
                 rate: int = 1) -> pd.DataFrame:
    def row_rate(row):
        row = row.copy()
        x = row[result_attr][::rate]
        t = np.array(x[time_attr])
        dt = t[1::] - t[:-1:]
        y = np.array(x[y_label])
        dy = y[1::] - y[:-1:]
        dy_dt = dy / dt
        row[result_attr] = pd.concat([pd.Series(t[1::]).rename(time_attr),
                                      pd.Series(dy_dt).rename(dy_dt_label or y_label)], axis=1)
        return row

    return df.apply(row_rate, axis=1)


def time_plot(df: pd.DataFrame, y_col: str, label_func: Callable,
              t_scale: number_t = 1.0 / 60.0, y_scale: number_t = 1, rate: int = 1,
              result_attr: str = 'results', time_attr: str = 'time',
              ax: Optional[Axes] = None, spline: bool = True):
    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111)
    for color, (index, row) in zip(colors, df.iterrows()):
        x = row[result_attr].iloc[1::rate]
        t = x[time_attr] * t_scale
        y = x[y_col] * y_scale
        if spline:
            ax.plot(t, y, 'o', color=color, alpha=0.3, ms=8, zorder=1, mec='k')
            ts = np.linspace(t.min(), t.max(), 200)
            ys = csaps.csaps(t, y, ts, smooth=0.9)
            ax.plot(ts, ys, '-', color=color, lw=10, alpha=0.75, zorder=2, label=label_func(index, row))
            ax.plot(ts, ys, '-', color='k', lw=1, alpha=1, zorder=3)
        else:
            ax.plot(t, y, 'o', color=color, alpha=0.7, ms=8, zorder=1,
                    label=label_func(index, row))
            ax.plot(t, y, '-', color='k', lw=1, alpha=1, zorder=3)


def create_mask(df: pd.DataFrame, attr: str, expr) -> Optional[pd.Series]:
    if expr is None:
        return None
    if isinstance(expr, (list, set, tuple)):
        return df[attr].isin(expr)
    return cast(pd.Series, df[attr] == expr)


def combine_masks(masks: Iterable[Optional[pd.Series]]):
    masks = filter(partial(operator.is_not, None), masks)
    if not masks:
        return None
    return reduce(operator.and_, masks)


def read_tdigest(binary_tdigest: Union[str, bytes]) -> TDigest:
    if isinstance(binary_tdigest, str):
        binary_tdigest = base64.b64decode(binary_tdigest)
    tdigest_encoding, = struct.unpack_from('>i', binary_tdigest)
    assert tdigest_encoding == 2
    t_min, t_max, t_compression, size, buffer_size, centroids = struct.unpack_from('>ddfhhh', binary_tdigest, 4)
    td = TDigest()
    for i in range(centroids):
        weight, mean = struct.unpack_from('>ff', binary_tdigest, 30 + i * 8)
        td.update(mean, weight)
    td.compress()
    return td


def tdigest_merge(t_digests: List[TDigest]) -> TDigest:
    first = t_digests[0]
    merged = TDigest(first.delta, first.K)
    for t_digest in t_digests:
        for c in t_digest.C.values():
            merged.update(c.mean, c.count)
    return merged


def compute_centroids(tdigest: TDigest) -> pd.DataFrame:
    centroids = pd.DataFrame(tdigest.centroids_to_list())
    centroids['cum_frac'] = centroids['c'].cumsum() / centroids['c'].sum()
    return centroids


def plot_centroids_cdf(centroids: pd.DataFrame, name: str, ax: Axes, **kwds):
    n = int(centroids["c"].sum())
    m = centroids['m'][centroids['cum_frac'] >= 0.5].iloc[0]
    centroids.plot('m', 'cum_frac', ax=ax,
                   label=f'{name:10} ({format_number(n)}, {format_number(m)})', **kwds)


def to_html(df: pd.DataFrame, index=False, **kwds) -> str:
    return ' '.join(df.to_html(index=index, **kwds).split())


def savefig(name, figdir, fig=None):
    (fig or plt.gcf()).savefig(f'{figdir}/{name}.png', bbox_inches='tight')


class ProgressFile:
    fp: FileIO
    progress: tqdm

    def __init__(self, fp, progress):
        self.fp = fp
        self.progress = progress

    def read(self, n):
        b = self.fp.read(n)
        self.progress.update(len(b))
        return b

    def readline(self):
        l = self.fp.readline()
        self.progress.update(len(l))
        return l

    def readinto(self, buff):
        n = self.fp.readinto(buff)
        self.progress.update(n)
        return n

    def close(self):
        self.fp.close()
        self.progress.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def open_progress(path: str, mode: str = 't') -> ProgressFile:
    return ProgressFile(open(path, 'r' + mode), tqdm(total=os.path.getsize(path), desc=path, unit_scale=True, unit='B'))


def load_pickle(path: str):
    with open_progress(path, 'b') as fp:
        return pickle.load(fp)

