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

def test_symm_s(lib):
    print("\nsymm type S (float)")
    
    lib.cblas_ssymm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]
    lib.cblas_ssymm.restype = None

    m, n = 3, 2
    alpha = 1.0
    beta = 1.0

    np.random.seed(42)
    A = np.random.rand(m, m).astype(np.float32)
    B = np.random.rand(m, n).astype(np.float32)
    C = np.eye(m, n, dtype=np.float32)

    lib.cblas_ssymm(
        CblasRowMajor, CblasLeft, CblasUpper,
        m, n,
        alpha,
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), m,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n,
        beta,
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n
    )

    print(f"Result C_s\n{C}")

def test_symm_d(lib):
    print("\nsymm type D (double)")
    
    lib.cblas_dsymm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int
    ]
    lib.cblas_dsymm.restype = None

    m, n = 3, 2
    np.random.seed(42)
    A = np.random.rand(m, m).astype(np.float64)
    B = np.random.rand(m, n).astype(np.float64)
    C = np.eye(m, n, dtype=np.float64)

    lib.cblas_dsymm(
        CblasRowMajor, CblasLeft, CblasUpper,
        m, n, 1.0,
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        1.0,
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n
    )
    print(f"Result C_d:\n{C}")

def test_symm_c(lib):
    print("\nsymm type C (complex)")
    
    lib.cblas_csymm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]
    lib.cblas_csymm.restype = None

    m, n = 3, 2
    np.random.seed(42)
    A = (np.random.rand(m, m) + 1j * np.random.rand(m, m)).astype(np.complex64)
    B = (np.random.rand(m, n) + 1j * np.random.rand(m, n)).astype(np.complex64)
    C = np.eye(m, n, dtype=np.complex64)

    alpha = (ctypes.c_float * 2)(1.0, 0.0)
    beta  = (ctypes.c_float * 2)(1.0, 0.0)

    lib.cblas_csymm(
        CblasRowMajor, CblasLeft, CblasUpper,
        m, n,
        ctypes.byref(alpha),
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), m,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n,
        ctypes.byref(beta),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n
    )
    print(f"Result C_c:\n{C}")

def test_symm_z(lib):
    print("\nsymm type Z (double complex)")
    
    lib.cblas_zsymm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int
    ]
    lib.cblas_zsymm.restype = None

    m, n = 3, 2
    np.random.seed(42)
    A = (np.random.rand(m, m) + 1j * np.random.rand(m, m)).astype(np.complex128)
    B = (np.random.rand(m, n) + 1j * np.random.rand(m, n)).astype(np.complex128)
    C = np.eye(m, n, dtype=np.complex128)

    alpha = (ctypes.c_double * 2)(1.0, 0.0)
    beta  = (ctypes.c_double * 2)(1.0, 0.0)

    lib.cblas_zsymm(
        CblasRowMajor, CblasLeft, CblasUpper,
        m, n,
        ctypes.byref(alpha),
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m,
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
        ctypes.byref(beta),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n
    )
    print(f"Result C_z:\n{C}")