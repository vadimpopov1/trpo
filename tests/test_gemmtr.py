import ctypes
import numpy as np
import os

CblasRowMajor = 101
CblasColMajor = 102
CblasNoTrans = 111
CblasTrans = 112
CblasConjTrans = 113
CblasUpper = 121
CblasLower = 122

def test_gemmtr_s(lib):
    print("\ngemmtr type S (float)")
    
    lib.cblas_sgemmt.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]
    lib.cblas_sgemmt.restype = None

    n, k = 3, 2
    alpha = 1.0
    beta = 1.0

    np.random.seed(42)
    A = np.random.rand(n, k).astype(np.float32)
    B = np.random.rand(k, n).astype(np.float32)
    C = np.eye(n, dtype=np.float32)

    lib.cblas_sgemmt(
        CblasRowMajor,
        CblasUpper,
        CblasNoTrans,
        CblasNoTrans,
        n, k,
        alpha,
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), k,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n,
        beta,
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n
    )

    print(f"Result C_s\n{C}")

def test_gemmtr_d(lib):
    print("\ngemmtr type D (double)")
    
    lib.cblas_dgemmt.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int
    ]
    lib.cblas_dgemmt.restype = None

    n, k = 3, 2
    np.random.seed(42)
    A = np.random.rand(n, k).astype(np.float64)
    B = np.random.rand(k, n).astype(np.float64)
    C = np.eye(n, dtype=np.float64)

    lib.cblas_dgemmt(
        CblasRowMajor, CblasUpper, CblasNoTrans, CblasNoTrans,
        n, k, 1.0,
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), k,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        1.0,
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n
    )
    print(f"Result C_d:\n{C}")

def test_gemmtr_c(lib):
    print("\ngemmtr type C (complex)")
    
    lib.cblas_cgemmt.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]
    lib.cblas_cgemmt.restype = None

    n, k = 3, 2
    np.random.seed(42)
    A = (np.random.rand(n, k) + 1j * np.random.rand(n, k)).astype(np.complex64)
    B = (np.random.rand(k, n) + 1j * np.random.rand(k, n)).astype(np.complex64)
    C = np.eye(n, dtype=np.complex64)

    alpha = (ctypes.c_float * 2)(1.0, 0.0)
    beta  = (ctypes.c_float * 2)(1.0, 0.0)

    lib.cblas_cgemmt(
        CblasRowMajor, CblasUpper, CblasNoTrans, CblasNoTrans,
        n, k,
        ctypes.byref(alpha),
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), k,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n,
        ctypes.byref(beta),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n
    )
    print(f"Result C_c:\n{C}")

def test_gemmtr_z(lib):
    print("\ngemmtr type Z (double complex)")
    
    lib.cblas_zgemmt.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int
    ]
    lib.cblas_zgemmt.restype = None

    n, k = 3, 2
    np.random.seed(42)
    A = (np.random.rand(n, k) + 1j * np.random.rand(n, k)).astype(np.complex128)
    B = (np.random.rand(k, n) + 1j * np.random.rand(k, n)).astype(np.complex128)
    C = np.eye(n, dtype=np.complex128)

    alpha = (ctypes.c_double * 2)(1.0, 0.0)
    beta  = (ctypes.c_double * 2)(1.0, 0.0)

    lib.cblas_zgemmt(
        CblasRowMajor, CblasUpper, CblasNoTrans, CblasNoTrans,
        n, k,
        ctypes.byref(alpha),
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), k,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(beta),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n
    )
    print(f"Result C_z:\n{C}")