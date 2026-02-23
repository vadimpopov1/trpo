import ctypes
import numpy as np

CblasRowMajor = 101
CblasColMajor = 102
CblasNoTrans = 111
CblasTrans = 112
CblasConjTrans = 113
CblasUpper = 121
CblasLower = 122
CblasLeft = 141
CblasRight = 142

def test_hemm_c(lib):
    print("\nhemm type C (complex)")
    
    lib.cblas_chemm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]
    lib.cblas_chemm.restype = None

    m, n = 3, 2
    np.random.seed(42)
    A = (np.random.rand(m, m) + 1j * np.random.rand(m, m)).astype(np.complex64)
    B = (np.random.rand(m, n) + 1j * np.random.rand(m, n)).astype(np.complex64)
    C = np.eye(m, n, dtype=np.complex64)

    alpha = (ctypes.c_float * 2)(1.0, 0.0)
    beta  = (ctypes.c_float * 2)(1.0, 0.0)

    lib.cblas_chemm(
        CblasRowMajor, CblasLeft, CblasUpper,
        m, n,
        ctypes.byref(alpha),
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), m,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n,
        ctypes.byref(beta),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n
    )
    print(f"Result C_c:\n{C}")

def test_hemm_z(lib):
    print("\nhemm type Z (double complex)")
    
    lib.cblas_zhemm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int
    ]
    lib.cblas_zhemm.restype = None

    m, n = 3, 2
    np.random.seed(42)
    A = (np.random.rand(m, m) + 1j * np.random.rand(m, m)).astype(np.complex128)
    B = (np.random.rand(m, n) + 1j * np.random.rand(m, n)).astype(np.complex128)
    C = np.eye(m, n, dtype=np.complex128)

    alpha = (ctypes.c_double * 2)(1.0, 0.0)
    beta  = (ctypes.c_double * 2)(1.0, 0.0)

    lib.cblas_zhemm(
        CblasRowMajor, CblasLeft, CblasUpper,
        m, n,
        ctypes.byref(alpha),
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(beta),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n
    )
    print(f"Result C_z:\n{C}")