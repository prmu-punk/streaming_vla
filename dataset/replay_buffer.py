from __future__ import annotations

import math
import numbers
import os
from functools import cached_property
from typing import Dict, Optional, Union

import numcodecs
import numpy as np
import zarr


def check_chunks_compatible(chunks: tuple, shape: tuple) -> None:
    assert len(shape) == len(chunks)
    for c in chunks:
        assert isinstance(c, numbers.Integral)
        assert c > 0


def rechunk_recompress_array(
    group,
    name,
    chunks=None,
    chunk_length=None,
    compressor=None,
    tmp_key="_temp",
):
    old_arr = group[name]
    if chunks is None:
        if chunk_length is not None:
            chunks = (chunk_length,) + old_arr.chunks[1:]
        else:
            chunks = old_arr.chunks
    check_chunks_compatible(chunks, old_arr.shape)

    if compressor is None:
        compressor = old_arr.compressor

    if (chunks == old_arr.chunks) and (compressor == old_arr.compressor):
        return old_arr

    group.move(name, tmp_key)
    old_arr = group[tmp_key]
    zarr.copy(
        source=old_arr,
        dest=group,
        name=name,
        chunks=chunks,
        compressor=compressor,
    )
    del group[tmp_key]
    return group[name]


def get_optimal_chunks(shape, dtype, target_chunk_bytes=2e6, max_chunk_length=None):
    itemsize = np.dtype(dtype).itemsize
    rshape = list(shape[::-1])
    if max_chunk_length is not None:
        rshape[-1] = int(max_chunk_length)
    split_idx = len(shape) - 1
    for i in range(len(shape) - 1):
        this_chunk_bytes = itemsize * np.prod(rshape[:i])
        next_chunk_bytes = itemsize * np.prod(rshape[: i + 1])
        if this_chunk_bytes <= target_chunk_bytes and next_chunk_bytes > target_chunk_bytes:
            split_idx = i

    rchunks = rshape[:split_idx]
    item_chunk_bytes = itemsize * np.prod(rshape[:split_idx])
    this_max_chunk_length = rshape[split_idx]
    next_chunk_length = min(this_max_chunk_length, math.ceil(target_chunk_bytes / item_chunk_bytes))
    rchunks.append(next_chunk_length)
    len_diff = len(shape) - len(rchunks)
    rchunks.extend([1] * len_diff)
    return tuple(rchunks[::-1])


class ReplayBuffer:
    def __init__(self, root: Union[zarr.Group, Dict[str, dict]]):
        assert "data" in root
        assert "meta" in root
        assert "episode_ends" in root["meta"]
        for key, value in root["data"].items():
            assert value.shape[0] == root["meta"]["episode_ends"][-1]
        self.root = root

    @classmethod
    def create_empty_zarr(cls, storage=None, root=None):
        if root is None:
            if storage is None:
                storage = zarr.MemoryStore()
            root = zarr.group(store=storage)
        root.require_group("data", overwrite=False)
        meta = root.require_group("meta", overwrite=False)
        if "episode_ends" not in meta:
            meta.zeros("episode_ends", shape=(0,), dtype=np.int64, compressor=None, overwrite=False)
        return cls(root=root)

    @classmethod
    def create_empty_numpy(cls):
        root = {
            "data": dict(),
            "meta": {
                "episode_ends": np.zeros((0,), dtype=np.int64),
            },
        }
        return cls(root=root)

    @classmethod
    def create_from_group(cls, group, **kwargs):
        if "data" not in group:
            return cls.create_empty_zarr(root=group, **kwargs)
        return cls(root=group, **kwargs)

    @classmethod
    def create_from_path(cls, zarr_path, mode="r", **kwargs):
        group = zarr.open(os.path.expanduser(zarr_path), mode)
        return cls.create_from_group(group, **kwargs)

    @classmethod
    def copy_from_store(
        cls,
        src_store,
        store=None,
        keys=None,
        chunks: Dict[str, tuple] = dict(),
        compressors: Union[dict, str, numcodecs.abc.Codec] = dict(),
        if_exists="replace",
        **kwargs,
    ):
        src_root = zarr.group(src_store)
        root = None
        if store is None:
            meta = dict()
            for key, value in src_root["meta"].items():
                meta[key] = np.array(value) if len(value.shape) == 0 else value[:]

            if keys is None:
                keys = src_root["data"].keys()
            data = dict()
            for key in keys:
                data[key] = src_root["data"][key][:]

            root = {"meta": meta, "data": data}
        else:
            root = zarr.group(store=store)
            zarr.copy_store(source=src_store, dest=store, source_path="/meta", dest_path="/meta", if_exists=if_exists)
            data_group = root.create_group("data", overwrite=True)
            if keys is None:
                keys = src_root["data"].keys()
            for key in keys:
                value = src_root["data"][key]
                cks = cls._resolve_array_chunks(chunks=chunks, key=key, array=value)
                cpr = cls._resolve_array_compressor(compressors=compressors, key=key, array=value)
                if cks == value.chunks and cpr == value.compressor:
                    this_path = "/data/" + key
                    zarr.copy_store(
                        source=src_store,
                        dest=store,
                        source_path=this_path,
                        dest_path=this_path,
                        if_exists=if_exists,
                    )
                else:
                    zarr.copy(source=value, dest=data_group, name=key, chunks=cks, compressor=cpr, if_exists=if_exists)
        return cls(root=root)

    @classmethod
    def copy_from_path(
        cls,
        zarr_path,
        backend=None,
        store=None,
        keys=None,
        chunks: Dict[str, tuple] = dict(),
        compressors: Union[dict, str, numcodecs.abc.Codec] = dict(),
        if_exists="replace",
        **kwargs,
    ):
        if backend == "numpy":
            store = None
        group = zarr.open(os.path.expanduser(zarr_path), "r")
        return cls.copy_from_store(
            src_store=group.store,
            store=store,
            keys=keys,
            chunks=chunks,
            compressors=compressors,
            if_exists=if_exists,
            **kwargs,
        )

    def save_to_store(
        self,
        store,
        chunks: Optional[Dict[str, tuple]] = dict(),
        compressors: Union[str, numcodecs.abc.Codec, dict] = dict(),
        if_exists="replace",
        **kwargs,
    ):
        root = zarr.group(store)
        if self.backend == "zarr":
            zarr.copy_store(source=self.root.store, dest=store, source_path="/meta", dest_path="/meta", if_exists=if_exists)
        else:
            meta_group = root.create_group("meta", overwrite=True)
            for key, value in self.root["meta"].items():
                meta_group.array(name=key, data=value, shape=value.shape, chunks=value.shape)

        data_group = root.create_group("data", overwrite=True)
        for key, value in self.root["data"].items():
            cks = self._resolve_array_chunks(chunks=chunks, key=key, array=value)
            cpr = self._resolve_array_compressor(compressors=compressors, key=key, array=value)
            if isinstance(value, zarr.Array):
                if cks == value.chunks and cpr == value.compressor:
                    this_path = "/data/" + key
                    zarr.copy_store(
                        source=self.root.store,
                        dest=store,
                        source_path=this_path,
                        dest_path=this_path,
                        if_exists=if_exists,
                    )
                else:
                    zarr.copy(source=value, dest=data_group, name=key, chunks=cks, compressor=cpr, if_exists=if_exists)
            else:
                data_group.array(name=key, data=value, chunks=cks, compressor=cpr)
        return store

    def save_to_path(
        self,
        zarr_path,
        chunks: Optional[Dict[str, tuple]] = dict(),
        compressors: Union[str, numcodecs.abc.Codec, dict] = dict(),
        if_exists="replace",
        **kwargs,
    ):
        store = zarr.DirectoryStore(os.path.expanduser(zarr_path))
        return self.save_to_store(store, chunks=chunks, compressors=compressors, if_exists=if_exists, **kwargs)

    @staticmethod
    def resolve_compressor(compressor="default"):
        if compressor == "default":
            compressor = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.NOSHUFFLE)
        elif compressor == "disk":
            compressor = numcodecs.Blosc("zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
        return compressor

    @classmethod
    def _resolve_array_compressor(cls, compressors: Union[dict, str, numcodecs.abc.Codec], key, array):
        cpr = "nil"
        if isinstance(compressors, dict):
            if key in compressors:
                cpr = cls.resolve_compressor(compressors[key])
            elif isinstance(array, zarr.Array):
                cpr = array.compressor
        else:
            cpr = cls.resolve_compressor(compressors)
        if cpr == "nil":
            cpr = cls.resolve_compressor("default")
        return cpr

    @classmethod
    def _resolve_array_chunks(cls, chunks: Union[dict, tuple], key, array):
        cks = None
        if isinstance(chunks, dict):
            if key in chunks:
                cks = chunks[key]
            elif isinstance(array, zarr.Array):
                cks = array.chunks
        elif isinstance(chunks, tuple):
            cks = chunks
        else:
            raise TypeError(f"Unsupported chunks type {type(chunks)}")
        if cks is None:
            cks = get_optimal_chunks(shape=array.shape, dtype=array.dtype)
        check_chunks_compatible(chunks=cks, shape=array.shape)
        return cks

    @cached_property
    def data(self):
        return self.root["data"]

    @cached_property
    def meta(self):
        return self.root["meta"]

    def update_meta(self, data):
        np_data = dict()
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                np_data[key] = value
            else:
                arr = np.array(value)
                if arr.dtype == object:
                    raise TypeError(f"Invalid value type {type(value)}")
                np_data[key] = arr

        meta_group = self.meta
        if self.backend == "zarr":
            for key, value in np_data.items():
                meta_group.array(name=key, data=value, shape=value.shape, chunks=value.shape, overwrite=True)
        else:
            meta_group.update(np_data)
        return meta_group

    @property
    def episode_ends(self):
        return self.meta["episode_ends"]

    @property
    def backend(self):
        return "zarr" if isinstance(self.root, zarr.Group) else "numpy"

    def __repr__(self) -> str:
        return str(self.root.tree()) if self.backend == "zarr" else super().__repr__()

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    @property
    def n_steps(self):
        if len(self.episode_ends) == 0:
            return 0
        return self.episode_ends[-1]

    @property
    def n_episodes(self):
        return len(self.episode_ends)

    @property
    def chunk_size(self):
        if self.backend == "zarr":
            return next(iter(self.data.arrays()))[-1].chunks[0]
        return None

    @property
    def episode_lengths(self):
        ends = self.episode_ends[:]
        ends = np.insert(ends, 0, 0)
        return np.diff(ends)
