#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

const int N = 100; // Размерность матрицы

// Умножение блока матрицы A на блок матрицы B
void blockMultiply(const vector<vector<double>>& A, const vector<vector<double>>& B, vector<vector<double>>& C, int startRowA, int endRowA) {
    for (int i = startRowA; i < endRowA; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i - startRowA][j] = 0;
            for (int k = 0; k < N; ++k) {
                C[i - startRowA][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, numProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    // Вычисляем количество блоков
    int numBlocks = N / numProcesses;

    // Инициализация матриц только на процессе с рангом 0
    vector<vector<double>> A(N, vector<double>(N));
    vector<vector<double>> B(N, vector<double>(N));
    vector<vector<double>> C(numBlocks, vector<double>(N)); // Результат для каждого процесса

    if (rank == 0) {
        // Заполняем матрицы случайными числами от 0 до 9
        srand(time(NULL));
        for (int i = 0; i < numBlocks; ++i) { // Изменили i < N на i < numBlocks
            for (int j = 0; j < N; ++j) { // Оставили без изменений
                C[i][j] = 0; // Добавили инициализацию матрицы C
            }
        }
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

    // Рассылаем матрицу B всем процессам
    MPI_Bcast(&B[0][0], N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Вычисляем блоки и умножаем их
    int startRowA = rank * numBlocks;
    int endRowA = (rank == numProcesses - 1) ? N : startRowA + numBlocks;
    blockMultiply(A, B, C, startRowA, endRowA);

    // Собираем результаты на процессе с рангом 0
    vector<vector<double>> globalC(N, vector<double>(N));
    MPI_Gather(&C[0][0], numBlocks * N, MPI_DOUBLE, &globalC[startRowA][0], numBlocks * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto endTimeMPI = high_resolution_clock::now();
    auto mpiTime = duration_cast<milliseconds>(endTimeMPI - startTimeMPI);

    // Вывод времени
    if (rank == 0) {
        cout << "Time: " << mpiTime.count() / 1000.0 << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}
