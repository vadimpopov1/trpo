import ctypes
import numpy as np
import os
from tests.test_gemm import *
from tests.test_gemmtr import *

CblasRowMajor = 101
CblasColMajor = 102
CblasNoTrans = 111
CblasTrans = 112
CblasConjTrans = 113
CblasUpper = 121
CblasLower = 122

def load_openblas():
    lib_path = os.path.abspath("third_party/OpenBLAS/libopenblas_armv8p-r0.3.31.dev.dylib")
    lib = ctypes.CDLL(lib_path)
    print("Library loaded from:", lib_path)
    return lib

if __name__ == "__main__":
    lib = load_openblas()
    test_gemm_s(lib)
    test_gemm_d(lib)
    test_gemm_c(lib)
    test_gemm_z(lib)
    test_gemmtr_s(lib)
    test_gemmtr_d(lib)
    test_gemmtr_c(lib)
    test_gemmtr_z(lib)