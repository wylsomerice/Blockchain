#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

const int N = 4; 
const int elementsCount = N * N;
//int B[N][N], A[N][N];
//int C[N][N];
void setMatrix(int* matrix) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[i * N + j] = rand() % 10;
        }
    }
}

int main(int argc, char* argv[]) {
    int i, j, k, rank, size, sum = 0;
    //int localA[N], localC[N];
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(time(nullptr) + rank);

    int* A = new int[elementsCount];
    int* B = new int[elementsCount];
    int* C = new int[elementsCount];

    if (rank == 0)
    {
        setMatrix(A);   

        setMatrix(B);
    }


    MPI_Bcast(B, elementsCount, MPI_INT, 0, MPI_COMM_WORLD);

    int blockSize = N / size;
    int* localA = new int[blockSize * N];
    MPI_Scatter(A, blockSize * N, MPI_INT, localA, blockSize * N, MPI_INT, 0, MPI_COMM_WORLD);

    int* localC = new int[blockSize * N];

    for (int i = 0; i < blockSize; ++i) {
        for (int j = 0; j < N; ++j) {
            localC[i * N + j] = 0;
            for (int k = 0; k < N; ++k) {
                localC[i * N + j] += localA[i * N + k] * B[k * N + j];
            }
        }
    }

    MPI_Gather(localC, blockSize * N, MPI_INT, C, blockSize * N, MPI_INT, 0, MPI_COMM_WORLD);


    //for (int i = 0; i < N / size; i++) {
    //    for (int j = 0; j < N; j++) {
    //        
    //        for (int k = 0; k < N; k++) {
    //            sum += localA[i * N + k] * B[k][j];
    //        }
    //        localC[i * N + j] = sum;
    //        sum = 0;
    //    }
    //}


    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Gather(localC, N * N / size, MPI_INT, C, N * N / size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {

        cout << "Matrix A:" << endl;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                cout << A[i * N + j] << " ";
            }
            cout << endl;
        }

        cout << "Matrix B:" << endl;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                cout << B[i * N + j] << " ";
            }
            cout << endl;
        }

        cout << "Result:" << endl;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                cout << C[i * N + j] << " ";
            }
            cout << endl;
        }
    }

    MPI_Finalize();
}
