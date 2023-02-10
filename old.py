import typing as tp

import numpy as np
import galois as gl
from itertools import combinations
from tqdm import tqdm

CONFIG = {
    'roots_max_count_ratio': 0.5,
    'gen_steps_limit': 64,
    'seed': 1111,
}


class RootsCountExceeded(RuntimeError):
    pass


class GenStepsExceeded(RuntimeError):
    pass


def get_roots(vs: np.ndarray, coords: tp.List[int], field) -> tp.Set[int]:
    kernel = vs[coords].null_space()
    rand_ind = np.random.choice(kernel.shape[0])
    normal = kernel[rand_ind]
    roots = gl.Poly(np.flip(normal), field=field).roots().flatten()

    return set(map(lambda x: int(x), roots))


def collect_roots(vs: np.ndarray, h: int, level: int, field) -> tp.Set:
    all_span_coords = list(combinations(range(h), level - 1))
    all_roots = set()

    for coords in all_span_coords:
        roots = get_roots(vs, list(coords), field)
        all_roots.update(roots)

    return all_roots


def gen_x(roots: tp.Set[int], field):
    for _ in range(CONFIG['gen_steps_limit']):
        elem = np.random.choice(field.order)
        if elem not in roots:
            return field(elem)

    raise GenStepsExceeded(
        f'Gen steps limit exceeded {CONFIG["gen_steps_limit"]}')


def build_vs(levels: np.ndarray, field_size) -> np.ndarray:
    field = gl.GF(field_size)
    max_level = levels[-1]
    vs = field.Zeros((levels.shape[0], max_level))

    for h in tqdm(range(levels.shape[0])):
        if h == 0:
            vs[0] = field([1 if i == 0 else 0 for i in range(max_level)])
            continue

        level = levels[h]

        roots = collect_roots(vs, h, level, field)
        roots.add(0)
        roots_count = len(roots)

        if roots_count >= CONFIG['roots_max_count_ratio'] * field_size:
            raise RootsCountExceeded(
                f'Too small field of size {field_size}. Roots collected {roots_count}')

        # Throws RuntimeError if number of steps exceeds corresp. parameter in config
        x = gen_x(roots, field)

        vs[h] = field([1 if i == 0 else 0 for i in range(max_level)])
        aux = x
        for i in range(1, max_level):
            vs[h, i] = aux
            aux *= x

    return vs


if __name__ == '__main__':
    np.random.seed(CONFIG.get('seed'))

    print(build_vs(levels=np.array(
        [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10]), field_size=2**18))
