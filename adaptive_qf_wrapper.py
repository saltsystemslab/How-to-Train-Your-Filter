import ctypes
import os

# load the shared library
lib_path = os.path.join(os.path.dirname(__file__), 'adaptiveqf', 'libadaptive_qf.so')
if not os.path.exists(lib_path):
    raise FileNotFoundError(
        f"Library not found at {lib_path}. "
        f"Run 'cd adaptiveqf && make python' first."
    )

lib = ctypes.CDLL(lib_path)

# AdaptiveQF class wrapper
class AdaptiveQF:
    def __init__(self, initial_size: int, max_size: int, fingerprint_size: int):
        self.obj = lib.adaptive_qf_create(initial_size, max_size, fingerprint_size)

    def insert(self, item: bytes) -> bool:
        return lib.adaptive_qf_insert(self.obj, item) == 1

    def contains(self, item: bytes) -> bool:
        return lib.adaptive_qf_contains(self.obj, item) == 1

    def delete(self, item: bytes) -> bool:
        return lib.adaptive_qf_delete(self.obj, item) == 1

    def get_size(self) -> int:
        return lib.adaptive_qf_get_size(self.obj)

    def get_capacity(self) -> int:
        return lib.adaptive_qf_get_capacity(self.obj)

    def __del__(self):
        lib.adaptive_qf_destroy(self.obj)

# utility functions

def set_insert():
    pass

def set_query():
    pass

def set_delete():
    pass

def set_free():
    pass

def insert_key():
    pass

def get_aqf_size(qf) -> int:
    return lib.get_aqf_size(qf.obj)
