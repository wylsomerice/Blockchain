#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

const int N = 100; 
int B[N][N], A[N][N];
int C[N][N];
void setMatrix(int matrix[N][N]) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[i][j] = rand() % 10 - 1;
        }
    }
}

int main(int argc, char* argv[]) {
    int i, j, k, rank, size, sum = 0;
    int localA[N], localC[N];
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        setMatrix(A);     
    }

    MPI_Scatter(A, N * N / size, MPI_INT, localA, N * N / size, MPI_INT, 0, MPI_COMM_WORLD);//Распределяем каждому процессу по одной строке матрицы
    MPI_Bcast(B, N * N, MPI_INT, 0, MPI_COMM_WORLD);//Распределяем каждому процессу по одной строке матрицы

    auto start = high_resolution_clock::now();

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            sum = sum + localA[j] * B[j][i];
        }
        localC[i] = sum;
        sum = 0;
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start).count();

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Gather(localC, N * N / size, MPI_INT, C, N * N / size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)//Вывод финальной матрицы после синхронизации и сбора данных 
    {
        cout << endl << endl << "Time with MPI: " << duration << endl << endl;
    }

    MPI_Finalize();
}
