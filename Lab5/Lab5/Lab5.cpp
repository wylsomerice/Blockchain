#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

const int N = 2000; 

void forwardElimination(vector<vector<double>>& A, vector<double>& b, int startRow, int endRow) {
    for (int k = 0; k < N - 1; ++k) {
        for (int i = startRow; i < endRow; ++i) {
            double factor = A[i][k] / A[k][k];
            for (int j = k; j < N; ++j) {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
        }
    }
}

void backwardSubstitution(vector<vector<double>>& A, vector<double>& b, vector<double>& x, int startRow, int endRow) {
    for (int k = N - 1; k >= 0; --k) {
        for (int i = startRow; i < endRow; ++i) {
            x[i] = b[i];
            for (int j = k + 1; j < N; ++j) {
                x[i] -= A[i][j] * x[j];
            }
            x[i] /= A[i][k];
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, numProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    int rowsPerProcess = N / numProcesses;
    int startRow = rank * rowsPerProcess;
    int endRow = (rank + 1) * rowsPerProcess;
    if (rank == numProcesses - 1) {
        endRow = N; 
    }

    vector<vector<double>> A(N, vector<double>(N));
    vector<double> b(N);

    vector<double> x(N);

    srand(time(NULL));
    for (int i = 0; i < N; ++i) {
        b[i] = rand() % 10; 
        for (int j = 0; j < N; ++j) {
            A[i][j] = rand() % 10; 
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto startTimeMPI = high_resolution_clock::now();

    forwardElimination(A, b, startRow, endRow);

    backwardSubstitution(A, b, x, startRow, endRow);

    auto endTimeMPI = high_resolution_clock::now();
    auto mpiTime = duration_cast<milliseconds>(endTimeMPI - startTimeMPI);

    vector<double> globalX(N);
    MPI_Gather(&x[0], rowsPerProcess, MPI_DOUBLE, &globalX[0], rowsPerProcess, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Solution:" << endl;
        for (int i = 0; i < N; ++i) {
            cout << "x[" << i << "] = " << globalX[i] << endl;
        }
        cout << "Time: " << mpiTime.count() / 1000.0 << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}
