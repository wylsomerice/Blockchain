#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;
using namespace chrono;

const double EPSILON = 1e-10;

void printMatrix(const vector<vector<double>>& matrix) {
    int n = matrix.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n + 1; ++j) {
            cout << matrix[i][j] << "\t";
        }
        cout << endl;
    }
}

vector<double> gaussEliminationSequential(vector<vector<double>>& matrix) {
    int n = matrix.size();

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double ratio = matrix[j][i] / matrix[i][i];
            for (int k = i; k < n + 1; ++k) {
                matrix[j][k] -= ratio * matrix[i][k];
            }
        }
    }

    vector<double> solution(n);
    for (int i = n - 1; i >= 0; --i) {
        solution[i] = matrix[i][n];
        for (int j = i + 1; j < n; ++j) {
            solution[i] -= matrix[i][j] * solution[j];
        }
        solution[i] /= matrix[i][i];
    }

    return solution;
}

vector<double> gaussEliminationParallel(vector<vector<double>>& matrix) {
    int n = matrix.size();

    for (int i = 0; i < n; ++i) {
#pragma omp parallel for
        for (int j = i + 1; j < n; ++j) {
            double ratio = matrix[j][i] / matrix[i][i];
            for (int k = i; k < n + 1; ++k) {
                matrix[j][k] -= ratio * matrix[i][k];
            }
        }
    }

    vector<double> solution(n);
    for (int i = n - 1; i >= 0; --i) {
        solution[i] = matrix[i][n];
#pragma omp parallel for
        for (int j = i + 1; j < n; ++j) {
            solution[i] -= matrix[i][j] * solution[j];
        }
        solution[i] /= matrix[i][i];
    }

    return solution;
}

int main() {

    setlocale(LC_ALL, "RUSSIAN");
    int n;
    cout << "Введите размерность матрицы: ";
    cin >> n;

    vector<vector<double>> matrix(n, vector<double>(n + 1));

    srand(time(0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n + 1; ++j) {
            matrix[i][j] = rand() % 10 + 1; 
        }
    }

    cout << "Исходная матрица:" << endl;
    printMatrix(matrix);
    cout << endl;

    auto start = high_resolution_clock::now();
    vector<double> solutionSeq = gaussEliminationSequential(matrix);
    auto stop = high_resolution_clock::now();
    auto durationSeq = duration_cast<microseconds>(stop - start);

    cout << "Время выполнения (без OpenMP): " << durationSeq.count() << " микросекунд" << endl;

    start = high_resolution_clock::now();
    vector<double> solutionPar = gaussEliminationParallel(matrix);
    stop = high_resolution_clock::now();
    auto durationPar = duration_cast<microseconds>(stop - start);

    cout << "Время выполнения (с OpenMP): " << durationPar.count() << " микросекунд" << endl;

    cout << "Решение СЛАУ:" << endl;
    cout << "Без OpenMP:" << endl;
    for (int i = 0; i < n; ++i) {
        cout << "x" << i + 1 << " = " << solutionSeq[i] << endl;
    }
    cout << "С OpenMP:" << endl;
    for (int i = 0; i < n; ++i) {
        cout << "x" << i + 1 << " = " << solutionPar[i] << endl;
    }

    return 0;
}
