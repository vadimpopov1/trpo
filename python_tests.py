import ctypes
import numpy as np
import os

def load_openblas():
    lib_path = os.path.abspath("third_party/OpenBLAS/libopenblas_armv8p-r0.3.31.dev.dylib")
    lib = ctypes.CDLL(lib_path)
    print("Library loaded from:", lib_path)
    return lib

def test_sgemm(lib):
    print("\ngemm type S (float):")
    lib.cblas_sgemm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]
    lib.cblas_sgemm.restype = None

    CblasRowMajor = 101
    CblasNoTrans = 111

    m, n, k = 2, 3, 2
    alpha = 1.0
    beta = 1.0

    np.random.seed(42)
    A = np.random.rand(m, k).astype(np.float32)
    B = np.random.rand(k, n).astype(np.float32)
    C = np.eye(m, n, dtype=np.float32)

    lib.cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        m, n, k,
        alpha,
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), k,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n,
        beta,
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n
    )
    print("Result C_s:\n", C)

def test_dgemm(lib):
    print("\ngemm type D (double)")
    lib.cblas_dgemm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int
    ]
    lib.cblas_dgemm.restype = None

    CblasRowMajor = 101
    CblasNoTrans = 111

    m, n, k = 2, 3, 2
    np.random.seed(42)
    A = np.random.rand(m, k).astype(np.float64)
    B = np.random.rand(k, n).astype(np.float64)
    C = np.eye(m, n, dtype=np.float64)

    lib.cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        m, n, k,
        1.0,
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), k,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        1.0,
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n
    )
    print("Result C_d:\n", C)

def test_cgemm(lib):
    print("\ngemm type C (complex)")
    lib.cblas_cgemm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]
    lib.cblas_cgemm.restype = None

    CblasRowMajor = 101
    CblasNoTrans = 111
    m, n, k = 2, 3, 2

    np.random.seed(42)
    A = (np.random.rand(m, k) + 1j * np.random.rand(m, k)).astype(np.complex64)
    B = (np.random.rand(k, n) + 1j * np.random.rand(k, n)).astype(np.complex64)
    C = np.eye(m, n, dtype=np.complex64)

    alpha = (ctypes.c_float * 2)(1.0, 0.0)
    beta  = (ctypes.c_float * 2)(1.0, 0.0)

    lib.cblas_cgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        m, n, k,
        ctypes.byref(alpha),
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), k,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n,
        ctypes.byref(beta),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n
    )
    print("Result C_c:\n", C)

def test_zgemm(lib):
    print("\ngemm type Z (double complex)")
    lib.cblas_zgemm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int
    ]
    lib.cblas_zgemm.restype = None

    CblasRowMajor = 101
    CblasNoTrans = 111
    m, n, k = 2, 3, 2

    np.random.seed(42)
    A = (np.random.rand(m, k) + 1j * np.random.rand(m, k)).astype(np.complex128)
    B = (np.random.rand(k, n) + 1j * np.random.rand(k, n)).astype(np.complex128)
    C = np.eye(m, n, dtype=np.complex128)

    alpha = (ctypes.c_double * 2)(1.0, 0.0)
    beta  = (ctypes.c_double * 2)(1.0, 0.0)

    lib.cblas_zgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        m, n, k,
        ctypes.byref(alpha),
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), k,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(beta),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n
    )
    print("Result C_z:\n", C)

if __name__ == "__main__":
    lib = load_openblas()
    test_sgemm(lib)
    test_dgemm(lib)
    test_cgemm(lib)
    test_zgemm(lib)