""" Dealer entity """

import typing as tp

import numpy as np
import galois as gl
from itertools import combinations
from tqdm import tqdm
from multiprocessing import Process, Queue


class RootsCountExceeded(RuntimeError):
    """ Throws when count of collected roots is greater or equal to field size """


def _get_roots_batch(span: np.ndarray, field) -> tp.Set[int]:
    """

    :param span:
    :param field:
    :return:
    """

    kernel = span.null_space()

    if kernel.size == 0:
        return set(map(lambda x: int(x), field.Elements()))

    roots = None
    for normal in kernel:
        roots_batch = set(
            map(lambda x: int(x),
                gl.Poly(np.flip(normal), field=field).roots().flatten()))
        if roots is None:
            roots = roots_batch
        else:
            roots = roots.intersection(roots_batch)

    return roots or set()


def _is_minimal_set(indices: np.ndarray):
    """

    :param indices:
    :return:
    """

    row = np.array([i for i in range(1, indices.shape[0] + 1)])
    res = (row - np.sort(indices)) >= 0
    return np.any(res)


def _collect_roots(results: Queue, sharing_matrix: np.ndarray,
                   indices_set: np.ndarray,
                   roots_cache: tp.Dict[str, tp.Set[int]], field) -> None:
    """

    :param results:
    :param sharing_matrix:
    :param indices_set:
    :param roots_cache:
    :param field:
    """

    local_roots_cache = {}
    roots: tp.Set[int] = set()

    for indices in indices_set:
        if not _is_minimal_set(indices):
            continue

        indices_hash = str(indices)
        if indices_hash in roots_cache:
            roots.update(roots_cache[indices_hash])
        else:
            roots_batch = _get_roots_batch(sharing_matrix[list(indices)],
                                           field)
            local_roots_cache[indices_hash] = roots_batch
            roots.update(roots_batch)

    results.put({'roots': roots})
    results.put({'cache': local_roots_cache})


class Dealer:
    """ Dealer class """

    def __init__(self, field_size: int, levels: np.ndarray):
        self.__field = gl.GF(field_size)
        self.__field_size: int = field_size
        self.__field_elems: tp.Set[int] = set([i for i in range(field_size)])

        self.__levels = levels
        self.__max_level = levels[-1]
        self.__sharing_matrix = self.__field.Zeros(
            (levels.shape[0] + 1, self.__max_level))
        self.__CONFIG = {
            'multiproc_using_threshold': 100,
            'max_proc_number': 8,
            'backtracking_size': 5,
            'backtracking_tries': 0,
        }
        self.__roots_cache: tp.Dict[str, tp.Set[int]] = {}
        self.__cache_snapshots: np.ndarray = np.empty(self.__levels.shape[0],
                                                      dtype=dict)
        self.__max_roots_count: int = 0

    def __collect_all_roots(self, h: int, level: int) -> tp.Set:
        """

        :param h:
        :param level:
        :return:
        """

        all_span_indices = np.array(list(combinations(range(h), level - 1)))
        results: Queue = Queue()
        procs = []
        proc_number = self.__CONFIG[
            'max_proc_number'] if all_span_indices.shape[0] >= self.__CONFIG[
                'multiproc_using_threshold'] else 1
        all_span_indices_chunks = np.array_split(all_span_indices, proc_number)

        for indices_set in all_span_indices_chunks:
            p = Process(target=_collect_roots,
                        args=(results, self.__sharing_matrix[:, :level],
                              indices_set, self.__roots_cache, self.__field))
            p.start()
            procs.append(p)

        all_roots: tp.Set[int] = set()
        for _ in range(2 * len(procs)):
            result = results.get()
            if 'cache' in result:
                self.__roots_cache.update(result['cache'])
            else:
                all_roots.update(result['roots'])

        for p in procs:
            p.join()

        return all_roots

    def __gen_free_elem(self, roots: tp.Set[int]):
        """

        :param roots:
        :return:
        """

        free_elem_candidates = np.array(list(self.__field_elems - roots))
        return self.__field(np.random.choice(free_elem_candidates))

    def __build_sharing_matrix_step(self, h: int) -> int:
        """

        :param h:
        :return:
        """

        # self.__cache_snapshots[h - 1] = self.__roots_cache

        level = self.__levels[h - 1]
        roots = self.__collect_all_roots(h, level)
        roots.add(0)
        roots_count = len(roots)

        if roots_count >= self.__field_size:
            return roots_count

        self.__max_roots_count = max(self.__max_roots_count, roots_count)
        free_elem = self.__gen_free_elem(roots)
        self.__sharing_matrix[h] = self.__field(
            [1 if i == 0 else 0 for i in range(self.__max_level)])

        aux = free_elem
        for i in range(1, level):
            self.__sharing_matrix[h, i] = aux
            aux = aux * free_elem

        return roots_count

    def __do_backtracking(self, hr: int) -> bool:
        """

        :param hr:
        :return:
        """

        hl = max(1, hr - self.__CONFIG['backtracking_size'])

        for _ in range(self.__CONFIG['backtracking_tries']):
            self.__roots_cache = self.__cache_snapshots[hl - 1]
            overflow_detected = False

            for h in range(hl, hr + 1):
                current_roots_count = self.__build_sharing_matrix_step(h)
                if current_roots_count >= self.__field_size:
                    overflow_detected = True
                    break

            if not overflow_detected:
                return True

        return False

    def build_sharing_matrix(self) -> tp.Tuple[int, np.ndarray]:
        self.__sharing_matrix[0] = self.__field(
            [1 if i == 0 else 0 for i in range(self.__max_level)])

        h = 1
        while h <= self.__levels.shape[0]:
            current_roots_count = self.__build_sharing_matrix_step(h)

            if current_roots_count >= self.__field_size:
                print(f'Backtracking started on iteration #{h}')
                backtracking_verdict = False  # self.__do_backtracking(h)
                if not backtracking_verdict:
                    raise RootsCountExceeded(
                        f'Too small field of size {self.__field_size}. Roots collected {current_roots_count}'
                    )
            h += 1

        return self.__max_roots_count, self.__sharing_matrix
