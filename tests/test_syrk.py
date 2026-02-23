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

def test_syrk_s(lib):
    print("\nsyrk type S (float)")
    
    lib.cblas_ssyrk.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]
    lib.cblas_ssyrk.restype = None

    n, k = 3, 2
    alpha = 1.0
    beta = 1.0

    np.random.seed(42)
    A = np.random.rand(n, k).astype(np.float32)
    C = np.eye(n, dtype=np.float32)

    lib.cblas_ssyrk(
        CblasRowMajor, CblasUpper, CblasNoTrans,
        n, k,
        alpha,
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), k,
        beta,
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n
    )

    print(f"Result C_s\n{C}")

def test_syrk_d(lib):
    print("\nsyrk type D (double)")
    
    lib.cblas_dsyrk.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int
    ]
    lib.cblas_dsyrk.restype = None

    n, k = 3, 2
    np.random.seed(42)
    A = np.random.rand(n, k).astype(np.float64)
    C = np.eye(n, dtype=np.float64)

    lib.cblas_dsyrk(
        CblasRowMajor, CblasUpper, CblasNoTrans,
        n, k, 1.0,
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), k,
        1.0,
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n
    )
    print(f"Result C_d:\n{C}")

def test_syrk_c(lib):
    print("\nsyrk type C (complex)")
    
    lib.cblas_csyrk.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]
    lib.cblas_csyrk.restype = None

    n, k = 3, 2
    np.random.seed(42)
    A = (np.random.rand(n, k) + 1j * np.random.rand(n, k)).astype(np.complex64)
    C = np.eye(n, dtype=np.complex64)

    alpha = (ctypes.c_float * 2)(1.0, 0.0)
    beta  = (ctypes.c_float * 2)(1.0, 0.0)

    lib.cblas_csyrk(
        CblasRowMajor, CblasUpper, CblasNoTrans,
        n, k,
        ctypes.byref(alpha),
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), k,
        ctypes.byref(beta),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n
    )
    print(f"Result C_c:\n{C}")

def test_syrk_z(lib):
    print("\nsyrk type Z (double complex)")
    
    lib.cblas_zsyrk.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int
    ]
    lib.cblas_zsyrk.restype = None

    n, k = 3, 2
    np.random.seed(42)
    A = (np.random.rand(n, k) + 1j * np.random.rand(n, k)).astype(np.complex128)
    C = np.eye(n, dtype=np.complex128)

    alpha = (ctypes.c_double * 2)(1.0, 0.0)
    beta  = (ctypes.c_double * 2)(1.0, 0.0)

    lib.cblas_zsyrk(
        CblasRowMajor, CblasUpper, CblasNoTrans,
        n, k,
        ctypes.byref(alpha),
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), k,
        ctypes.byref(beta),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n
    )
    print(f"Result C_z:\n{C}")