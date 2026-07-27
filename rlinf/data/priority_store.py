# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from sortedcontainers import SortedList

from rlinf.data.embodied_io_struct import Trajectory


class PriorityStore:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self._seq = 0
        # Sort by (priority, seq) so that among equal-priority items the oldest
        # (lowest seq) is always at index 0 and evicted first.
        # priority is a tuple (min_version, mean_version) for lexicographic ordering:
        # first prefer higher min_version, then break ties by higher mean_version.
        self.sl = SortedList(key=lambda x: (x[0], x[1]))

        # Tracking for never-used trajectory count.
        self._used_seqs: set = set()
        self._discarded_unused: int = 0

    def add(self, priority: tuple[float, float], data: Trajectory) -> bool:
        if len(self.sl) == self.maxsize:
            if priority < self.sl[0][0]:
                self._discarded_unused += 1
                return False

        self.sl.add((priority, self._seq, data))
        self._seq += 1

        if len(self.sl) > self.maxsize:
            evicted = self.sl.pop(0)
            evicted_seq = evicted[1]
            if evicted_seq not in self._used_seqs:
                self._discarded_unused += 1
            else:
                self._used_seqs.discard(evicted_seq)

        return True

    def topn(self, n: int) -> list[Trajectory]:
        items = self.sl[-n:]
        for _, seq, _ in items:
            self._used_seqs.add(seq)
        return list(reversed([data for _, _, data in items]))

    def peek_topn(self, n: int) -> list[Trajectory]:
        """View the top-n highest-priority candidates without marking or removing.

        Used by the chunk_mask readiness collective to inspect candidates across
        all ranks before deciding to take or discard them synchronously.
        """
        items = self.sl[-n:]
        return list(reversed([data for _, _, data in items]))

    def take_topn(self, n: int) -> list[Trajectory]:
        """Remove and return the top-n highest-priority candidates, marking them used.

        Guarantees a completed episode is trained on at most once per global step:
        consumed candidates are removed from the store and their seqs are marked
        used so that later eviction does not double-count them as discarded-unused.
        """
        items = self.sl[-n:]
        for item in items:
            self._used_seqs.add(item[1])
            self.sl.remove(item)
        return list(reversed([data for _, _, data in items]))

    def discard_topn(self, n: int) -> None:
        """Discard the top-n highest-priority candidates without marking them used.

        Called by every rank in lockstep when the readiness collective observes
        that some rank has zero fresh chunks: all ranks drop their current
        candidates so fresh data can refill the store. Never-used discards
        increment ``_discarded_unused`` (matching ``remove_below`` semantics).
        """
        items = self.sl[-n:]
        for item in items:
            seq = item[1]
            if seq not in self._used_seqs:
                self._discarded_unused += 1
            else:
                self._used_seqs.discard(seq)
            self.sl.remove(item)

    def remove_below(self, threshold):
        to_remove = [item for item in self.sl if item[0][0] < threshold]
        for item in to_remove:
            seq = item[1]
            if seq not in self._used_seqs:
                self._discarded_unused += 1
            else:
                self._used_seqs.discard(seq)
            self.sl.remove(item)

    def get_metric(self) -> dict:
        total_cells = 0
        counts: dict = {}

        for _, _, data in self.sl:
            if data.versions is None:
                continue
            flat = torch.round(data.versions.reshape(-1)).to(torch.int64)
            uniq, cnt = torch.unique(flat, return_counts=True)
            for v, c in zip(uniq.tolist(), cnt.tolist()):
                counts[v] = counts.get(v, 0) + c
            total_cells += flat.numel()

        if total_cells == 0:
            return {"discarded_unused": self._discarded_unused}

        result = {v: {"ratio": c / total_cells} for v, c in counts.items()}
        result["discarded_unused"] = self._discarded_unused
        return result

    def __len__(self):
        return len(self.sl)
