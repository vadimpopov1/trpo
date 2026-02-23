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

def test_trmm_s(lib):
    print("\ntrmm type S (float)")
    
    lib.cblas_strmm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]
    lib.cblas_strmm.restype = None

    m, n = 3, 2
    alpha = 1.0

    np.random.seed(42)
    A = np.triu(np.random.rand(m, m).astype(np.float32))
    B = np.random.rand(m, n).astype(np.float32)

    lib.cblas_strmm(
        CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        m, n,
        alpha,
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), m,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n
    )

    print(f"Result B_s\n{B}")

def test_trmm_d(lib):
    print("\ntrmm type D (double)")
    
    lib.cblas_dtrmm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int
    ]
    lib.cblas_dtrmm.restype = None

    m, n = 3, 2
    np.random.seed(42)
    A = np.triu(np.random.rand(m, m).astype(np.float64))
    B = np.random.rand(m, n).astype(np.float64)

    lib.cblas_dtrmm(
        CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        m, n, 1.0,
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n
    )
    print(f"Result B_d:\n{B}")

def test_trmm_c(lib):
    print("\ntrmm type C (complex)")
    
    lib.cblas_ctrmm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]
    lib.cblas_ctrmm.restype = None

    m, n = 3, 2
    np.random.seed(42)
    A = np.triu((np.random.rand(m, m) + 1j * np.random.rand(m, m)).astype(np.complex64))
    B = (np.random.rand(m, n) + 1j * np.random.rand(m, n)).astype(np.complex64)

    alpha = (ctypes.c_float * 2)(1.0, 0.0)

    lib.cblas_ctrmm(
        CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        m, n,
        ctypes.byref(alpha),
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), m,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n
    )
    print(f"Result B_c:\n{B}")

def test_trmm_z(lib):
    print("\ntrmm type Z (double complex)")
    
    lib.cblas_ztrmm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int
    ]
    lib.cblas_ztrmm.restype = None

    m, n = 3, 2
    np.random.seed(42)
    A = np.triu((np.random.rand(m, m) + 1j * np.random.rand(m, m)).astype(np.complex128))
    B = (np.random.rand(m, n) + 1j * np.random.rand(m, n)).astype(np.complex128)

    alpha = (ctypes.c_double * 2)(1.0, 0.0)

    lib.cblas_ztrmm(
        CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        m, n,
        ctypes.byref(alpha),
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n
    )
    print(f"Result B_z:\n{B}")