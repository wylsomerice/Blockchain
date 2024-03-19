#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

const int N = 100; // Размерность матрицы

// Умножение части матрицы A на матрицу B
void matrixMultiply(const vector<vector<double>>& A, const vector<vector<double>>& B, vector<vector<double>>& C, int startRow, int endRow) {
    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, numProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    // Рассчитываем количество строк, которые будет обрабатывать каждый процесс
    int rowsPerProcess = N / numProcesses;
    int startRow = rank * rowsPerProcess;
    int endRow = (rank == numProcesses - 1) ? N : startRow + rowsPerProcess;

    // Инициализация матриц только на процессе с рангом 0
    vector<vector<double>> A(N, vector<double>(N));
    vector<vector<double>> B(N, vector<double>(N));
    vector<vector<double>> localA(endRow - startRow, vector<double>(N));
    vector<vector<double>> C(endRow - startRow, vector<double>(N));

    if (rank == 0) {
        // Заполняем матрицы случайными числами от 0 до 9
        srand(time(NULL));
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                A[i][j] = rand() % 10;
                B[i][j] = rand() % 10;
            }
        }
    }

    // Измеряем время для умножения матриц с использованием MPI
    MPI_Barrier(MPI_COMM_WORLD);
    auto startTimeMPI = high_resolution_clock::now();

    // Рассылаем части матрицы A всем процессам
    MPI_Bcast(&A[0][0], N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Распределяем часть матрицы A между процессами
    MPI_Scatter(&A[startRow][0], (endRow - startRow) * N, MPI_DOUBLE, &localA[0][0], (endRow - startRow) * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Умножение части матрицы A на матрицу B
    matrixMultiply(localA, B, C, 0, endRow - startRow);

    // Собираем результаты на процессе с рангом 0
    vector<vector<double>> globalC(N, vector<double>(N));
    MPI_Gather(&C[0][0], (endRow - startRow) * N, MPI_DOUBLE, &globalC[startRow][0], (endRow - startRow) * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto endTimeMPI = high_resolution_clock::now();
    auto mpiTime = duration_cast<milliseconds>(endTimeMPI - startTimeMPI);

    // Вывод времени
    if (rank == 0) {
        cout << "Time: " << mpiTime.count() / 1000.0 << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}
