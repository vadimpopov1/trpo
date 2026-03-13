import ctypes
import os
from tests.test_gemm import *
from tests.test_gemmtr import *
from tests.test_symn import *
from tests.test_hemm import *
from tests.test_trmm import *
from tests.test_trsm import *
from tests.test_syrk import *
from tests.test_herk import *
from tests.test_syr2k import *
from tests.test_her2k import *

def load_openblas():
    lib_path = os.path.abspath("third_party/OpenBLAS/libopenblas_armv8p-r0.3.31.dev.dylib")
    lib = ctypes.CDLL(lib_path)
    print("Library loaded from:", lib_path)
    return lib

if __name__ == "__main__":
    lib = load_openblas()
    try:
        test_gemm_d(lib)
        print("PASSED")
    except:
        print("FAILED")
    
    test_gemm_c(lib)
    test_gemm_z(lib)

    test_gemmtr_s(lib)
    test_gemmtr_d(lib)
    test_gemmtr_c(lib)
    test_gemmtr_z(lib)

    test_symm_s(lib)
    test_symm_d(lib)
    test_symm_c(lib)
    test_symm_z(lib)

    test_hemm_c(lib)
    test_hemm_z(lib)

    test_trmm_s(lib)
    test_trmm_d(lib)
    test_trmm_c(lib)
    test_trmm_z(lib)

    test_trsm_s(lib)
    test_trsm_d(lib)
    test_trsm_c(lib)
    test_trsm_z(lib)

    test_syrk_s(lib)
    test_syrk_d(lib)
    test_syrk_c(lib)
    test_syrk_z(lib)

    test_herk_c(lib)
    test_herk_z(lib)

    test_syr2k_s(lib)
    test_syr2k_d(lib)
    test_syr2k_c(lib)
    test_syr2k_z(lib)

    test_her2k_c(lib)
    test_her2k_z(lib)