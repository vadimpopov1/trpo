#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <dlfcn.h>

using namespace std;
using namespace std::chrono;

typedef void (*ssyrk_func)(int, int, int, int, int, float, float*, int, float, float*, int);

void ssyrk_simple(char uplo, char trans, int n, int k, float alpha, float* A, int lda, float beta, float* C, int ldc) {
    if (n <= 0 || k < 0) return;
    
    bool upper = (uplo == 'U' || uplo == 'u');
    bool normal = (trans == 'N' || trans == 'n');
    
    if (normal) {
        if (upper) {
            for (int j = 0; j < n; j++) {
                for (int i = 0; i <= j; i++) {
                    float sum = 0;
                    for (int l = 0; l < k; l++) {
                        sum += A[i + l*lda] * A[j + l*lda];
                    }
                    if (beta == 0)
                        C[i + j*ldc] = alpha * sum;
                    else
                        C[i + j*ldc] = alpha * sum + beta * C[i + j*ldc];
                }
            }
        } else {
            for (int j = 0; j < n; j++) {
                for (int i = j; i < n; i++) {
                    float sum = 0;
                    for (int l = 0; l < k; l++) {
                        sum += A[i + l*lda] * A[j + l*lda];
                    }
                    if (beta == 0)
                        C[i + j*ldc] = alpha * sum;
                    else
                        C[i + j*ldc] = alpha * sum + beta * C[i + j*ldc];
                }
            }
        }
    } else {
        if (upper) {
            for (int j = 0; j < n; j++) {
                for (int i = 0; i <= j; i++) {
                    float sum = 0;
                    for (int l = 0; l < k; l++) {
                        sum += A[l + i*lda] * A[l + j*lda];
                    }
                    if (beta == 0)
                        C[i + j*ldc] = alpha * sum;
                    else
                        C[i + j*ldc] = alpha * sum + beta * C[i + j*ldc];
                }
            }
        } else {
            for (int j = 0; j < n; j++) {
                for (int i = j; i < n; i++) {
                    float sum = 0;
                    for (int l = 0; l < k; l++) {
                        sum += A[l + i*lda] * A[l + j*lda];
                    }
                    if (beta == 0)
                        C[i + j*ldc] = alpha * sum;
                    else
                        C[i + j*ldc] = alpha * sum + beta * C[i + j*ldc];
                }
            }
        }
    }
}

int main() {
    int n = 1024;
    int k = 512;
    
    int lda = n;
    int ldc = n;
    
    vector<float> A(lda * k);
    vector<float> C1(ldc * n);
    vector<float> C2(ldc * n);
    
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int i = 0; i < lda * k; i++) A[i] = dist(gen);
    for (int i = 0; i < ldc * n; i++) {
        C1[i] = dist(gen);
        C2[i] = C1[i];
    }
    
    auto start = high_resolution_clock::now();
    ssyrk_simple('U', 'N', n, k, 1.0f, A.data(), lda, 0.5f, C1.data(), ldc);
    auto end = high_resolution_clock::now();
    double simple_time = duration_cast<microseconds>(end - start).count() / 1000000.0;
    cout << "Простая реализация: " << simple_time << " сек" << endl;
    
    void* handle = dlopen("../third_party/OpenBLAS/libopenblas.dylib", RTLD_LAZY);
    if (handle) {
        ssyrk_func cblas_ssyrk = (ssyrk_func)dlsym(handle, "cblas_ssyrk");
        
        if (cblas_ssyrk) {
            start = high_resolution_clock::now();
            cblas_ssyrk(101, 121, 111, n, k, 1.0f, A.data(), lda, 0.5f, C2.data(), ldc);
            end = high_resolution_clock::now();
            double blas_time = duration_cast<microseconds>(end - start).count() / 1000000.0;
            cout << "OpenBLAS: " << blas_time << " сек" << endl;
            
            float percent = (blas_time / simple_time) * 100.0f;
            cout << "OpenBLAS составляет " << percent << "% от скорости простой реализации" << endl;
        }
        dlclose(handle);
    }
    
    return 0;
}