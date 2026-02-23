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

def test_trsm_s(lib):
    print("\ntrsm type S (float)")
    
    lib.cblas_strsm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]
    lib.cblas_strsm.restype = None

    m, n = 3, 2
    alpha = 1.0

    np.random.seed(42)
    A = np.triu(np.random.rand(m, m).astype(np.float32))
    B = np.random.rand(m, n).astype(np.float32)

    lib.cblas_strsm(
        CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        m, n,
        alpha,
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), m,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n
    )

    print(f"Result B_s\n{B}")

def test_trsm_d(lib):
    print("\ntrsm type D (double)")
    
    lib.cblas_dtrsm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int
    ]
    lib.cblas_dtrsm.restype = None

    m, n = 3, 2
    np.random.seed(42)
    A = np.triu(np.random.rand(m, m).astype(np.float64))
    B = np.random.rand(m, n).astype(np.float64)

    lib.cblas_dtrsm(
        CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        m, n, 1.0,
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n
    )
    print(f"Result B_d:\n{B}")

def test_trsm_c(lib):
    print("\ntrsm type C (complex)")
    
    lib.cblas_ctrsm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]
    lib.cblas_ctrsm.restype = None

    m, n = 3, 2
    np.random.seed(42)
    A = np.triu((np.random.rand(m, m) + 1j * np.random.rand(m, m)).astype(np.complex64))
    B = (np.random.rand(m, n) + 1j * np.random.rand(m, n)).astype(np.complex64)

    alpha = (ctypes.c_float * 2)(1.0, 0.0)

    lib.cblas_ctrsm(
        CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        m, n,
        ctypes.byref(alpha),
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), m,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n
    )
    print(f"Result B_c:\n{B}")

def test_trsm_z(lib):
    print("\ntrsm type Z (double complex)")
    
    lib.cblas_ztrsm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int
    ]
    lib.cblas_ztrsm.restype = None

    m, n = 3, 2
    np.random.seed(42)
    A = np.triu((np.random.rand(m, m) + 1j * np.random.rand(m, m)).astype(np.complex128))
    B = (np.random.rand(m, n) + 1j * np.random.rand(m, n)).astype(np.complex128)

    alpha = (ctypes.c_double * 2)(1.0, 0.0)

    lib.cblas_ztrsm(
        CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        m, n,
        ctypes.byref(alpha),
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n
    )
    print(f"Result B_z:\n{B}")