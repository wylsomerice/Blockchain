#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Генерация случайных чисел от 0 до 9
int generateRandom() {
    return rand() % 10;
}

// Метод Гаусса для решения СЛАУ на части матрицы и вектора
void gaussElimination(vector<vector<double>>& A, vector<double>& b, int startRow, int endRow) {
    const int n = A.size();

    for (int i = 0; i < n; ++i) {
        double pivot = A[i][i];
        if (pivot == 0) {
            throw runtime_error("Matrix is singular");
        }

        for (int j = i + 1; j < n; ++j) {
            double ratio = A[j][i] / pivot;
            for (int k = i; k < n; ++k) {
                A[j][k] -= ratio * A[i][k];
            }
            b[j] -= ratio * b[i];
        }
    }

    // Обратный ход метода Гаусса
    for (int i = n - 1; i >= 0; --i) {
        b[i] /= A[i][i];
        A[i][i] = 1.0;

        for (int j = i - 1; j >= 0; --j) {
            b[j] -= A[j][i] * b[i];
            A[j][i] = 0.0;
        }
    }
}

int main(int argc, char** argv) {
    srand(time(NULL));
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 100; 

    const int rowsPerProcess = N / size;
    const int remainingRows = N % size;

    vector<vector<double>> localA(rowsPerProcess + (rank < remainingRows ? 1 : 0), vector<double>(N));
    vector<double> localB(rowsPerProcess + (rank < remainingRows ? 1 : 0));

    if (rank == 0) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                localA[i][j] = generateRandom();
            }
            localB[i] = generateRandom();
        }
    }

    MPI_Scatter(localA[0].data(), rowsPerProcess * N, MPI_DOUBLE, localA[0].data(), rowsPerProcess * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(localB.data(), rowsPerProcess, MPI_DOUBLE, localB.data(), rowsPerProcess, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto start = high_resolution_clock::now();
    gaussElimination(localA, localB, 0, localA.size());
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start).count();

    vector<double> result(N);
    MPI_Gather(localB.data(), rowsPerProcess, MPI_DOUBLE, result.data(), rowsPerProcess, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Solution:" << endl;
        for (int i = 0; i < N; ++i) {
            cout << result[i] << " ";
        }
        cout << endl;
        cout << "Time taken: " << duration << " milliseconds" << endl;
    }

    MPI_Finalize();
    return 0;
}