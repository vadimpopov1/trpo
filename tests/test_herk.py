import ctypes
import numpy as np

CblasRowMajor = 101
CblasColMajor = 102
CblasNoTrans = 111
CblasTrans = 112
CblasConjTrans = 113
CblasUpper = 121
CblasLower = 122
CblasNonUnit = 131
CblasUnit = 132
CblasLeft = 141
CblasRight = 142

def test_herk_c(lib):
    print("\nherk type C (complex)")
    
    lib.cblas_cherk.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]
    lib.cblas_cherk.restype = None

    n, k = 3, 2
    alpha = 1.0
    beta = 1.0

    np.random.seed(42)
    A = (np.random.rand(n, k) + 1j * np.random.rand(n, k)).astype(np.complex64)
    C = np.eye(n, dtype=np.complex64)

    lib.cblas_cherk(
        CblasRowMajor, CblasUpper, CblasNoTrans,
        n, k,
        alpha,
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), k,
        beta,
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n
    )
    print(f"Result C_c:\n{C}")

def test_herk_z(lib):
    print("\nherk type Z (double complex)")
    
    lib.cblas_zherk.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int
    ]
    lib.cblas_zherk.restype = None

    n, k = 3, 2
    alpha = 1.0
    beta = 1.0

    np.random.seed(42)
    A = (np.random.rand(n, k) + 1j * np.random.rand(n, k)).astype(np.complex128)
    C = np.eye(n, dtype=np.complex128)

    lib.cblas_zherk(
        CblasRowMajor, CblasUpper, CblasNoTrans,
        n, k,
        alpha,
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), k,
        beta,
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n
    )
    print(f"Result C_z:\n{C}")