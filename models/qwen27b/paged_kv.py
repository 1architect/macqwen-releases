#!/usr/bin/env python3
"""paged_kv.py - ContextVM S3: paged full-attention KV with exact merge.

Attention over pages never builds one contiguous K/V tensor. Each page
contributes partial softmax statistics. The statistics merge with the
numerically stable online-softmax rule from PROJECT.md section 27:

    m = max(m_old, m_new)
    l = exp(m_old - m) * l_old + exp(m_new - m) * l_new
    o = exp(m_old - m) * o_old + exp(m_new - m) * o_new
    out = o / l

Only the 16 full-attention layers use this. The 48 GDN layers keep their
fixed-size recurrent state.

Run the unit test with no model load:

    python3 paged_kv.py --selftest
"""

import argparse, itertools, os, shutil, stat, sys, tempfile, time
from pathlib import Path

import mlx.core as mx

from mlx_lm.models.cache import _BaseCache


def require_free_memory(min_gb=8.0):
    """Refuse to load the model unless the machine has room.

    The weights need about 11.25 GB. Starting with less free memory than that
    guarantees swap, which does not merely run slow: the process enters
    uninterruptible disk wait and the whole Mac becomes unusable.
    """
    import subprocess, re as _re
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int((_re.search(r"page size of (\d+)", out) or [0, 16384])[1])
    def pages(label):
        m = _re.search(rf"{label}:\s+(\d+)", out)
        return int(m.group(1)) if m else 0
    free = (pages("Pages free") + pages("Pages inactive")
            + pages("Pages purgeable")) * page / 1e9
    if free < min_gb:
        print(f"ABORT: only {free:.2f} GB free, need {min_gb:.1f} GB.")
        print("Close apps and retry. Loading now would swap and freeze the Mac.")
        sys.exit(2)
    print(f"preflight: {free:.2f} GB free, ok", flush=True)


_CACHE_IDS = itertools.count()


class PagedKVCache(_BaseCache):
    """Full-attention KV stored as fixed-size pages."""

    def __init__(self, page_size: int = 256, top_k_pages: int = 0,
                 pinned_pages: int = 1, recent_pages: int = 2,
                 refresh_every: int = 32, min_context: int = 16384,
                 spill_dir=None, resident_pages: int = 0):
        self.page_size = page_size
        self.k_pages = []      # each [B, n_kv, page_size, D], preallocated
        self.v_pages = []
        self.fill = []         # filled token count per page
        self.offset = 0
        # --- S4 sparse selection ---
        self.top_k_pages = top_k_pages     # 0 disables selection (attend to all)
        self.pinned_pages = pinned_pages   # first N pages always selected
        self.recent_pages = recent_pages   # last N pages always selected
        self.refresh_every = refresh_every
        self.min_context = min_context     # below this, selection is off
        self.gather_decode = True          # one cached gather + the growing page
        self.k_min = []        # per page [n_kv, D]
        self.k_max = []
        self._stack_dirty = True
        self._min_stack = None
        self._max_stack = None
        self._selection = None
        self._steps_since_refresh = 0
        self.selection_log = []
        # cached gather of the stable (non-growing) selected pages
        self._gk = None
        self._gv = None
        self._g_sig = None
        # --- S6 SSD page store ---
        # Cold pages leave RAM and live on disk. Only pinned, recent and
        # selected pages stay resident, so physical KV stops tracking context
        # length. The min/max bounds always stay in RAM: they are tiny
        # (33 MB for 256K tokens) and selection needs them for every page.
        # A caller may share one parent directory between layers and engine
        # instances. Each cache owns one private directory below that parent.
        # The private directory is created atomically, so processes cannot
        # collide on page names or remove another cache's pages.
        self._spill_parent = Path(spill_dir).expanduser() if spill_dir else None
        self.spill_dir = None
        self._spill_identity = None
        self.resident_pages = resident_pages   # 0 = never spill
        self._on_disk = set()
        self._cache_id = next(_CACHE_IDS)
        self.spill_stats = {"spilled": 0, "restored": 0, "bytes_out": 0, "bytes_in": 0}
        if self._spill_parent:
            self._spill_parent.mkdir(parents=True, exist_ok=True)
            self.spill_dir = Path(tempfile.mkdtemp(
                prefix=f"macqwen-pages-{os.getpid()}-",
                dir=str(self._spill_parent),
            ))
            identity = self.spill_dir.stat()
            self._spill_identity = (identity.st_dev, identity.st_ino)

    # -- cache protocol -----------------------------------------------------

    def update_and_fetch(self, keys, values):
        """Append new keys/values into preallocated pages.

        Growing the last page with mx.concatenate copies the whole page on
        every decode step. Preallocating and writing in place removes that.
        """
        B, n_kv, L, _ = keys.shape
        Dk, Dv = keys.shape[3], values.shape[3]
        pos = 0
        while pos < L:
            if not self.k_pages or self.fill[-1] == self.page_size:
                self.k_pages.append(mx.zeros((B, n_kv, self.page_size, Dk), keys.dtype))
                self.v_pages.append(mx.zeros((B, n_kv, self.page_size, Dv), values.dtype))
                self.fill.append(0)
            f = self.fill[-1]
            take = min(self.page_size - f, L - pos)
            self.k_pages[-1][..., f:f + take, :] = keys[:, :, pos:pos + take, :]
            self.v_pages[-1][..., f:f + take, :] = values[:, :, pos:pos + take, :]
            self.fill[-1] = f + take
            pos += take
        self.offset += L
        self._update_bounds()
        self._enforce_budget()
        return keys, values

    # -- S6: SSD page store -------------------------------------------------

    def _page_path(self, i):
        return self.spill_dir / f"c{self._cache_id:03d}_p{i:06d}.safetensors"

    def _spill(self, i):
        """Move one page out of RAM."""
        if i in self._on_disk or self.spill_dir is None:
            return
        k, v = self.k_pages[i], self.v_pages[i]
        if k is None:
            return
        path = self._page_path(i)
        # A restored page is unchanged, so its file is still valid. Rewriting
        # it doubles SSD writes for nothing.
        if not path.exists():
            mx.eval(k, v)
            mx.save_safetensors(str(path), {"k": k, "v": v})
            self.spill_stats["bytes_out"] += k.nbytes + v.nbytes
        self.spill_stats["spilled"] += 1
        self.k_pages[i] = None
        self.v_pages[i] = None
        self._on_disk.add(i)

    def _restore(self, i):
        """Bring one page back into RAM."""
        d = mx.load(str(self._page_path(i)))
        self.k_pages[i] = d["k"]
        self.v_pages[i] = d["v"]
        self._on_disk.discard(i)
        self.spill_stats["restored"] += 1
        self.spill_stats["bytes_in"] += d["k"].nbytes + d["v"].nbytes

    def _enforce_budget(self, keep=()):
        """Spill cold pages until the resident count fits the budget."""
        if self.spill_dir is None or self.resident_pages <= 0:
            return
        n = len(self.k_pages)
        protected = set(keep)
        protected |= set(range(min(self.pinned_pages, n)))
        protected |= set(range(max(0, n - self.recent_pages - 1), n))
        resident = [i for i in range(n) if self.k_pages[i] is not None]
        excess = len(resident) - self.resident_pages
        if excess <= 0:
            return
        # evict the oldest resident pages that nothing needs right now
        for i in resident:
            if excess <= 0:
                break
            if i in protected:
                continue
            self._spill(i)
            excess -= 1

    def close(self):
        path = self.spill_dir
        identity = self._spill_identity
        if path is None or identity is None:
            return
        # Do not follow a replacement symlink or delete a directory that was
        # created after this cache closed. The parent is never removed.
        try:
            current = path.lstat()
            if not stat.S_ISDIR(current.st_mode):
                return
            if (current.st_dev, current.st_ino) != identity:
                return
            shutil.rmtree(path, ignore_errors=True)
        except FileNotFoundError:
            return

    def kv(self, i):
        """Filled slice of page i, restoring it from disk if needed."""
        if i in self._on_disk:
            self._restore(i)
        f = self.fill[i]
        if f == self.page_size:
            return self.k_pages[i], self.v_pages[i]
        return self.k_pages[i][..., :f, :], self.v_pages[i][..., :f, :]

    # -- S4: per-page key bounds -------------------------------------------

    def _update_bounds(self):
        """Recompute min/max bounds for pages that changed."""
        while len(self.k_min) < len(self.k_pages):
            self.k_min.append(None)
            self.k_max.append(None)
        # a bulk prefill creates many pages at once, so refresh every page
        # that has no bounds yet, plus the last page because it can grow
        todo = [i for i, m in enumerate(self.k_min)
                if m is None and self.k_pages[i] is not None]
        if self.k_pages and self.k_pages[-1] is not None:
            last = len(self.k_pages) - 1
            if last not in todo:
                todo.append(last)
        for i in todo:
            kp = self.kv(i)[0][0]                   # [n_kv, filled, D]
            self.k_min[i] = mx.min(kp, axis=1)      # [n_kv, D]
            self.k_max[i] = mx.max(kp, axis=1)
        self._stack_dirty = True

    def _stacks(self):
        if self._stack_dirty or self._min_stack is None:
            self._min_stack = mx.stack(self.k_min)   # [P, n_kv, D]
            self._max_stack = mx.stack(self.k_max)
            self._stack_dirty = False
        return self._min_stack, self._max_stack

    def page_scores(self, queries):
        """Admissible upper bound of q.k for every page.

        For each dimension the largest possible product is q_d * max_d when
        q_d > 0 and q_d * min_d otherwise. Summing those gives an upper bound
        on the true dot product, so a page can never be wrongly ranked below
        its real relevance.
        """
        mn, mx_ = self._stacks()
        n_pages, n_kv, D = mn.shape
        q = queries[0, :, -1, :]                     # [n_q, D]
        n_rep = q.shape[0] // n_kv
        qh = q.reshape(n_kv, n_rep, D)
        qp = mx.maximum(qh, 0)
        qn = mx.minimum(qh, 0)
        best = None
        for j in range(n_kv):
            # [n_rep, D] @ [D, P] -> [n_rep, P]
            u = qp[j] @ mx_[:, j, :].T + qn[j] @ mn[:, j, :].T
            m = mx.max(u, axis=0)                    # [P]
            best = m if best is None else mx.maximum(best, m)
        return best

    def select_pages(self, queries):
        """Return the page indices to attend to for this decode step."""
        n_pages = len(self.k_pages)
        if (self.top_k_pages <= 0 or n_pages <= self.top_k_pages
                or self.offset < self.min_context):
            # Below min_context the KV is a small share of the bytes read per
            # token, so skipping pages cannot pay for its own overhead.
            return None                              # attend to everything
        if (self._selection is not None
                and self._steps_since_refresh < self.refresh_every):
            self._steps_since_refresh += 1
            return self._selection

        forced = set(range(min(self.pinned_pages, n_pages)))
        forced |= set(range(max(0, n_pages - self.recent_pages), n_pages))
        budget = self.top_k_pages - len(forced)
        if budget <= 0:
            chosen = sorted(forced)[: self.top_k_pages]
        else:
            scores = self.page_scores(queries)
            order = mx.argsort(-scores).tolist()
            chosen = set(forced)
            for idx in order:
                if len(chosen) >= self.top_k_pages:
                    break
                chosen.add(int(idx))
            chosen = sorted(chosen)
        self._selection = chosen
        self._steps_since_refresh = 1
        self.selection_log.append((n_pages, len(chosen)))
        return chosen

    @property
    def state(self):
        return [self.kv(i)[j] for j in (0, 1) for i in range(len(self.k_pages))]

    @state.setter
    def state(self, v):
        if v is None:
            v = []
        if not isinstance(v, (list, tuple)) or len(v) % 2:
            raise ValueError("PagedKVCache state must contain K/V pairs")
        n = len(v) // 2
        source_k, source_v = list(v[:n]), list(v[n:])
        if any(k is None or val is None for k, val in zip(source_k, source_v)):
            raise ValueError("PagedKVCache state cannot contain empty pages")
        self.fill = [p.shape[2] for p in source_k]
        if any(fill > self.page_size for fill in self.fill):
            raise ValueError("PagedKVCache state page exceeds page_size")

        def pad(page, fill):
            if fill == self.page_size:
                return page
            shape = page.shape[:2] + (self.page_size,) + page.shape[3:]
            padded = mx.zeros(shape, page.dtype)
            padded[..., :fill, :] = page
            return padded

        self.k_pages = [pad(page, fill) for page, fill in zip(source_k, self.fill)]
        self.v_pages = [pad(page, fill) for page, fill in zip(source_v, self.fill)]
        self.offset = sum(self.fill)
        self._on_disk.clear()
        self.k_min = [None] * n
        self.k_max = [None] * n
        self._stack_dirty = True
        self._min_stack = None
        self._max_stack = None
        self._selection = None
        self._steps_since_refresh = 0
        self.selection_log = []
        self._gk = None
        self._gv = None
        self._g_sig = None

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        left = self.offset - n
        self.offset = left
        keep_k, keep_v, keep_f, seen = [], [], [], 0
        for i in range(len(self.k_pages)):
            if seen >= left:
                break
            take = min(self.fill[i], left - seen)
            keep_k.append(self.k_pages[i])
            keep_v.append(self.v_pages[i])
            keep_f.append(take)
            seen += take
        self.k_pages, self.v_pages, self.fill = keep_k, keep_v, keep_f
        self._g_sig = None
        return n

    def empty(self):
        return self.offset == 0

    @property
    def nbytes(self):
        """Resident bytes only. Spilled pages do not count."""
        return sum(p.nbytes for p in self.k_pages if p is not None) + \
               sum(p.nbytes for p in self.v_pages if p is not None)

    @property
    def resident_count(self):
        return sum(1 for p in self.k_pages if p is not None)

    def stable_gather(self, idx):
        """Contiguous K/V for the selected full pages, cached across refreshes.

        Rebuilding this every decode step copies the whole selection, which
        costs more bandwidth than the sparsity saves. The growing last page is
        excluded and handled separately, so this only changes on a refresh.
        """
        sig = tuple(idx)
        if self._g_sig != sig:
            self._enforce_budget(keep=idx)
            ks = [self.kv(i)[0] for i in idx]
            vs = [self.kv(i)[1] for i in idx]
            self._gk = mx.concatenate(ks, axis=2) if ks else None
            self._gv = mx.concatenate(vs, axis=2) if vs else None
            self._g_sig = sig
        return self._gk, self._gv

    def page_bounds(self, only=None):
        """Yield (page_index, start, end) in logical token positions."""
        start = 0
        for i in range(len(self.k_pages)):
            end = start + self.fill[i]
            if only is None or i in only:
                yield i, start, end
            start = end


def _page_mask(mask, q_len, total_len, start, end, n_repeats):
    """Slice a mask down to one page's key range."""
    if mask is None:
        return None
    if isinstance(mask, str):
        # causal: query i attends keys up to (total_len - q_len + i)
        q_idx = mx.arange(total_len - q_len, total_len)[:, None]
        k_idx = mx.arange(start, end)[None]
        return q_idx >= k_idx
    m = mask[..., start:end]
    if n_repeats > 1 and m.ndim > 3:
        m = mx.expand_dims(m, -3)
    return m


def paged_attention(queries, cache, scale, mask=None):
    """Exact attention over pages using online-softmax merging.

    Prefill (L > 1) uses the fused kernel over the concatenated pages. The
    per-page loop is only worth it at decode. With L = 3000 and 24 pages the
    loop builds 24 score tensors of [1, 4, 6, 3000, page] in float32 inside a
    single lazy graph, which exhausts memory on a 16 GB machine.
    """
    B, n_q_heads, L, D = queries.shape
    if L > 1:
        k = mx.concatenate([cache.kv(i)[0] for i in range(len(cache.k_pages))], axis=2)
        v = mx.concatenate([cache.kv(i)[1] for i in range(len(cache.k_pages))], axis=2)
        return mx.fast.scaled_dot_product_attention(
            queries, k, v, scale=scale, mask=mask)
    n_kv = cache.k_pages[0].shape[1]
    n_repeats = n_q_heads // n_kv
    total = cache.offset

    q = queries * scale
    if n_repeats > 1:
        q = q.reshape(B, n_kv, n_repeats, L, D)

    m_run = None   # running max
    l_run = None   # running exp-sum
    o_run = None   # running weighted value sum

    selected = None
    sel = cache.select_pages(queries)
    selected = set(sel) if sel is not None else None

    # Decode chunks to merge over. With gather_decode the selected full pages
    # are one cached contiguous buffer, rebuilt only when the selection
    # refreshes, plus the growing last page. That is two chunks instead of one
    # per page, and no per-step copy of the whole selection.
    if cache.gather_decode:
        idx = sorted(selected) if selected is not None else list(range(len(cache.k_pages)))
        last = len(cache.k_pages) - 1
        gk, gv = cache.stable_gather([i for i in idx if i != last])
        chunks = []
        if gk is not None:
            chunks.append((gk, gv))
        if last in idx:
            chunks.append(cache.kv(last))
    else:
        chunks = [cache.kv(i) for i, _, _ in cache.page_bounds(only=selected)]

    for k, v in chunks:
        if n_repeats > 1:
            k = mx.expand_dims(k, -3)
            v = mx.expand_dims(v, -3)

        # softmax statistics accumulate in float32; the fused kernel does the
        # same internally. bfloat16 accumulation costs ~6e-3 relative error,
        # which compounds across 16 attention layers into wrong logits.
        scores = (q @ mx.swapaxes(k, -1, -2)).astype(mx.float32)

        m_new = mx.max(scores, axis=-1, keepdims=True)
        p = mx.exp(scores - m_new)
        l_new = mx.sum(p, axis=-1, keepdims=True)
        o_new = (p @ v.astype(mx.float32))

        if m_run is None:
            m_run, l_run, o_run = m_new, l_new, o_new
        else:
            m_cat = mx.maximum(m_run, m_new)
            a = mx.exp(m_run - m_cat)
            b = mx.exp(m_new - m_cat)
            l_run = a * l_run + b * l_new
            o_run = a * o_run + b * o_new
            m_run = m_cat

    out = (o_run / l_run).astype(queries.dtype)
    if n_repeats > 1:
        out = out.reshape(B, n_q_heads, L, D)
    return out


# ----------------------------------------------------------------------------
# install into the model
# ----------------------------------------------------------------------------

def install():
    """Patch the Qwen attention entry point to dispatch paged caches."""
    import mlx_lm.models.qwen3_next as qn
    if getattr(qn, "_paged_installed", False):
        return
    original = qn.scaled_dot_product_attention

    def dispatch(queries, keys, values, cache, scale, mask=None, sinks=None):
        if isinstance(cache, PagedKVCache):
            return paged_attention(queries, cache, scale, mask)
        return original(queries, keys, values, cache=cache, scale=scale,
                        mask=mask, sinks=sinks)

    qn.scaled_dot_product_attention = dispatch
    qn._paged_installed = True


def make_paged_cache(model, page_size=256, **kwargs):
    """Like make_prompt_cache, but full-attention layers use pages."""
    from mlx_lm.models.cache import ArraysCache
    return [ArraysCache(size=2) if l.is_linear else PagedKVCache(page_size, **kwargs)
            for l in model.language_model.model.layers]


# ----------------------------------------------------------------------------
# unit test: paged versus fused kernel, no model needed
# ----------------------------------------------------------------------------

def selftest(page_size=256):
    """Paged attention must be at least as accurate as the fused kernel.

    In bfloat16 the fused kernel is itself an approximation. Comparing the two
    directly measures rounding, not correctness, so both are scored against a
    float32 reference.
    """
    mx.random.seed(0)
    B, n_q, n_kv, D = 1, 24, 4, 256
    scale = D ** -0.5
    ok = True
    for total, q_len, label in [(1000, 1, "decode"),
                                (1000, 37, "prefill chunk"),
                                (777, 777, "full prefill"),
                                (256, 1, "exact page boundary"),
                                (300, 1, "partial last page")]:
        k32 = mx.random.normal((B, n_kv, total, D))
        v32 = mx.random.normal((B, n_kv, total, D))
        q32 = mx.random.normal((B, n_q, q_len, D))
        mask = "causal" if q_len > 1 else None

        truth = mx.fast.scaled_dot_product_attention(q32, k32, v32, scale=scale, mask=mask)
        mx.eval(truth)
        norm = float(mx.max(mx.abs(truth)))

        q, k, v = (x.astype(mx.bfloat16) for x in (q32, k32, v32))
        fused = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        cache = PagedKVCache(page_size)
        cache.gather_decode = False
        cache.update_and_fetch(k, v)
        paged = paged_attention(q, cache, scale, mask)
        mx.eval(fused, paged)

        e_fused = float(mx.max(mx.abs(fused.astype(mx.float32) - truth))) / norm
        e_paged = float(mx.max(mx.abs(paged.astype(mx.float32) - truth))) / norm
        # bfloat16 noise floor is ~1e-2; both paths sit inside it
        good = e_paged < 1.5e-2 and e_paged <= max(e_fused * 3.0, 1e-3)
        ok = ok and good
        print(f"{label:<22} total={total:<5} q_len={q_len:<4} pages={len(cache.k_pages):<3} "
              f"fused_err={e_fused:.3e} paged_err={e_paged:.3e} {'PASS' if good else 'FAIL'}")
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def sparse_test(page_size=64, n_pages=40, top_k=6, needle_gain=8.0):
    """Diagnose top-k page selection.

    Reports how the min/max upper bound ranks pages against the true ranking
    by actual maximum dot product, the recall of the true top-k, and the
    attention error from reading only the selected pages.
    """
    mx.random.seed(0)
    B, n_q, n_kv, D = 1, 24, 4, 256
    scale = D ** -0.5
    total = n_pages * page_size

    k = mx.random.normal((B, n_kv, total, D)) * 0.35
    v = mx.random.normal((B, n_kv, total, D))
    q = mx.random.normal((B, n_q, 1, D))

    needle = n_pages // 3
    lo, hi = needle * page_size, (needle + 1) * page_size
    direction = q[0, :, 0, :].reshape(n_kv, n_q // n_kv, D).mean(axis=1)
    k[:, :, lo:hi, :] = direction[None, :, None, :] * (needle_gain / D ** 0.5)

    # true per-page relevance: max dot product over the page and heads
    qh = q[0, :, 0, :].reshape(n_kv, n_q // n_kv, D)
    true = []
    for i in range(n_pages):
        kp = k[0, :, i * page_size:(i + 1) * page_size, :]       # [n_kv, P, D]
        best = max(float(mx.max(qh[j] @ kp[j].T)) for j in range(n_kv))
        true.append(best)
    true_order = sorted(range(n_pages), key=lambda i: -true[i])
    true_topk = set(true_order[:top_k])

    cache = PagedKVCache(page_size, top_k_pages=top_k, pinned_pages=1,
                         recent_pages=1, refresh_every=1, min_context=0)
    cache.gather_decode = False
    cache.update_and_fetch(k, v)
    scores = cache.page_scores(q)
    mx.eval(scores)
    bound_order = mx.argsort(-scores).tolist()
    chosen = cache.select_pages(q)

    dense = PagedKVCache(page_size)
    dense.gather_decode = False
    dense.update_and_fetch(k, v)
    full = paged_attention(q, dense, scale, None)
    sparse = paged_attention(q, cache, scale, None)
    mx.eval(full, sparse)
    rel = float(mx.max(mx.abs(full - sparse))) / float(mx.max(mx.abs(full)))

    recall = len(true_topk & set(chosen)) / len(true_topk)
    print(f"pages={n_pages} page_size={page_size} top_k={top_k} needle_gain={needle_gain}")
    print(f"true best pages   : {true_order[:top_k]}")
    print(f"bound best pages  : {bound_order[:top_k]}")
    print(f"selected          : {chosen}")
    print(f"needle page {needle}: true_rank={true_order.index(needle)} "
          f"bound_rank={bound_order.index(needle)} selected={needle in chosen}")
    print(f"recall of true top-{top_k}: {recall:.2f}")
    print(f"KV not read       : {100*(1-len(chosen)/n_pages):.0f}%")
    print(f"sparse vs full    : rel={rel:.3e}")
    ok = recall >= 0.5 and needle in chosen
    print("SPARSE TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def needle_test(model_path, page_size=128, n_prompt=4000, depth=0.5,
                ks=(0,), source_root=None, n_gen=24):
    """Quality gate: can sparse selection still find one fact in a long context?

    A secret value is planted at a chosen depth inside real source text. The
    model is asked for it. k = 0 is dense attention, the control that shows
    the model can retrieve the fact at all.
    """
    from mlx_lm import load
    from mlx_lm.generate import wired_limit
    from mlx_lm.models.cache import make_prompt_cache

    SECRET = "74391"
    NEEDLE = (f"\n\n// IMPORTANT PROJECT FACT: the secret MacBat build number "
              f"is {SECRET}. Remember this exact number.\n\n")
    QUESTION = ("\n\nQuestion: what is the secret MacBat build number stated "
                "above? Answer with only the number.\nAnswer:")

    require_free_memory(8.0)
    print(f"loading {model_path}", flush=True)
    model, tokenizer = load(model_path)
    install()

    chunks = []
    for f in sorted(Path(source_root or Path.cwd()).rglob("*.swift")):
        if any(x in f.parts for x in (".build", "DerivedData", ".git")):
            continue
        chunks.append(f"// FILE: {f.name}\n" + f.read_text(errors="replace"))
        if sum(len(c) for c in chunks) > n_prompt * 4:
            break
    body = "\n\n".join(chunks)
    cut = int(len(body) * depth)
    text = body[:cut] + NEEDLE + body[cut:]
    ids = tokenizer.encode(text, add_special_tokens=False)[:n_prompt]
    ids += tokenizer.encode(QUESTION, add_special_tokens=False)
    prompt = mx.array(ids)[None]
    n_pages_est = (len(ids) + page_size - 1) // page_size
    print(f"context {len(ids)} tokens, page_size {page_size}, "
          f"~{n_pages_est} pages, needle at depth {depth:.0%}", flush=True)
    print(f"secret planted: {SECRET}\n", flush=True)

    results = []
    with wired_limit(model):
        for k in ks:
            if k == 0:
                cache = make_prompt_cache(model)
                label = "dense (control)"
            else:
                cache = make_paged_cache(model, page_size, top_k_pages=k,
                                         pinned_pages=1, recent_pages=2,
                                         refresh_every=8, min_context=0)
                label = f"sparse k={k}"
            t = time.perf_counter()
            step = 256
            for i in range(0, prompt.shape[1], step):
                logits = model(prompt[:, i:i + step], cache=cache)
                mx.eval([c.state for c in cache])
                mx.clear_cache()
            out = []
            y = mx.argmax(logits[0, -1]).item()
            out.append(y)
            for _ in range(n_gen - 1):
                logits = model(mx.array([[y]]), cache=cache)
                y = mx.argmax(logits[0, -1]).item()
                out.append(y)
                if y in tokenizer.eos_token_ids:
                    break
            ans = tokenizer.decode(out).strip()
            hit = SECRET in ans
            read = ""
            if k and isinstance(cache[3], PagedKVCache) and cache[3].selection_log:
                tot, sel = cache[3].selection_log[-1]
                read = f"  [{sel}/{tot} pages read, {100*(1-sel/tot):.0f}% skipped]"
            print(f"{label:<16} {time.perf_counter()-t:6.1f}s  "
                  f"{'FOUND ' if hit else 'MISSED'}  {ans[:60]!r}{read}", flush=True)
            results.append((label, hit))
            del cache
            mx.clear_cache()

    print()
    for label, hit in results:
        print(f"  {label:<16} {'PASS' if hit else 'FAIL'}")
    ok = all(h for _, h in results)
    print("NEEDLE TEST:", "PASS" if ok else "PARTIAL" if results[0][1] else "INVALID CONTROL")
    return 0 if ok else 1


def model_test(model_path, page_size=256, n_prompt=240, n_gen=48,
               top_k=0, source_root=None, baseline=True):
    """Acceptance: paged attention must generate the same tokens as contiguous."""
    from mlx_lm import load
    from mlx_lm.generate import wired_limit
    from mlx_lm.models.cache import make_prompt_cache

    print(f"loading {model_path}")
    t0 = time.perf_counter()
    model, tokenizer = load(model_path)
    print(f"loaded in {time.perf_counter()-t0:.1f}s")
    install()

    if source_root:
        from pathlib import Path
        chunks = []
        for f in sorted(Path(source_root).rglob("*.swift")):
            if any(x in f.parts for x in (".build", "DerivedData", ".git")):
                continue
            chunks.append(f"// FILE: {f.name}\n" + f.read_text(errors="replace"))
            if sum(len(c) for c in chunks) > n_prompt * 4:
                break
        text = "\n\n".join(chunks)
    else:
        text = ("The MacBat application manages battery state on macOS. "
                "Explain how the AppDelegate coordinates the menu bar item, "
                "the battery sampler, and the persistence layer. ") * 8
    ids = tokenizer.encode(text, add_special_tokens=False)[:n_prompt]
    prompt = mx.array(ids)[None]
    print(f"prompt tokens: {prompt.shape[1]}  page_size: {page_size}")

    def greedy(cache, label, step=256):
        t = time.perf_counter()
        # Chunked prefill. One unchunked forward pass of 1600 tokens keeps the
        # activations of all 64 layers in a single lazy graph, which peaked at
        # 14.3 GB and drove the machine into swap. stream_generate chunks and
        # evaluates between chunks for exactly this reason.
        n = prompt.shape[1]
        for i in range(0, n, step):
            logits = model(prompt[:, i:i + step], cache=cache)
            mx.eval([c.state for c in cache])
            mx.clear_cache()
        mx.eval(logits)
        prefill_s = time.perf_counter() - t
        print(f"{label:<11} prefill {prefill_s:6.1f}s ({n/prefill_s:5.1f} tok/s)  "
              f"mlx act {mx.get_active_memory()/1e9:5.2f} peak {mx.get_peak_memory()/1e9:5.2f} GB",
              flush=True)
        t = time.perf_counter()   # decode timing must exclude prefill
        out = []
        y = mx.argmax(logits[0, -1]).item()
        out.append(y)
        for _ in range(n_gen - 1):
            logits = model(mx.array([[y]]), cache=cache)
            y = mx.argmax(logits[0, -1]).item()
            out.append(y)
        dt = time.perf_counter() - t
        print(f"{label:<11} {n_gen} tokens {dt:5.1f}s ({n_gen/dt:4.2f} tok/s)  "
              f"mlx act {mx.get_active_memory()/1e9:5.2f} peak {mx.get_peak_memory()/1e9:5.2f} GB",
              flush=True)
        return out

    # wired_limit pins the working set. Without it the weights page in and out
    # on a 16 GB machine and a plain forward pass stalls in disk wait.
    with wired_limit(model):
        ref_tokens = greedy(make_prompt_cache(model), "contiguous") if baseline else None
        paged = make_paged_cache(model, page_size, top_k_pages=top_k,
                                 pinned_pages=1, recent_pages=2, refresh_every=8,
                                 min_context=0)
        got_tokens = greedy(paged, "paged" if not top_k else f"sparse k={top_k}")

    n_pages = len(paged[3].k_pages)
    if not baseline:
        if paged[3].selection_log:
            tot, sel = paged[3].selection_log[-1]
            print(f"selection: {sel}/{tot} pages read "
                  f"({100*(1-sel/tot):.0f}% of KV skipped)")
        print(f"pages: {n_pages}")
        print("output:", repr(tokenizer.decode(got_tokens)[:300]))
        print("MODEL TEST: sparse-only run, no baseline")
        return 0
    if top_k and paged[3].selection_log:
        tot, sel = paged[3].selection_log[-1]
        print(f"selection: {sel}/{tot} pages read "
              f"({100*(1-sel/tot):.0f}% of KV skipped), refreshes={len(paged[3].selection_log)}")
    match = sum(1 for a, b in zip(ref_tokens, got_tokens) if a == b)
    first_div = next((i for i, (a, b) in enumerate(zip(ref_tokens, got_tokens)) if a != b), None)
    print(f"full-attention pages: {n_pages}")
    print(f"identical tokens: {match}/{n_gen}"
          + (f", first divergence at {first_div}" if first_div is not None else ", no divergence"))
    print("contiguous:", repr(tokenizer.decode(ref_tokens)[:160]))
    print("paged     :", repr(tokenizer.decode(got_tokens)[:160]))

    ok = match == n_gen
    print("MODEL TEST:", "PASS" if ok else "PARTIAL" if match > n_gen * 0.8 else "FAIL")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--model-test", action="store_true")
    p.add_argument("--sparse-test", action="store_true")
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--needle-gain", type=float, default=8.0)
    p.add_argument("--top-k-pages", type=int, default=0)
    p.add_argument("--source-root", default=None)
    p.add_argument("--needle-test", action="store_true")
    p.add_argument("--depth", type=float, default=0.5)
    p.add_argument("--ks", default="0,4,2")
    p.add_argument("--no-baseline", action="store_true",
                   help="skip the contiguous run; test only the sparse path")
    p.add_argument(
        "--model",
        default=str(Path.home() / "models/Qwen3.8-27B-Apple-MLX-E2-v1"),
    )
    p.add_argument("--page-size", type=int, default=256)
    p.add_argument("--tokens", type=int, default=240)
    p.add_argument("--gen", type=int, default=48)
    a = p.parse_args()
    if a.selftest:
        return selftest(a.page_size)
    if a.sparse_test:
        return sparse_test(top_k=a.top_k, needle_gain=a.needle_gain)
    if a.needle_test:
        return needle_test(a.model, a.page_size, a.tokens, a.depth,
                           tuple(int(x) for x in a.ks.split(',')),
                           a.source_root, a.gen)
    if a.model_test:
        return model_test(a.model, a.page_size, a.tokens, a.gen,
                          top_k=a.top_k_pages, source_root=a.source_root,
                          baseline=not a.no_baseline)
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
