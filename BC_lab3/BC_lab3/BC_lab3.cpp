#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

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

vector<double> gaussElimination(vector<vector<double>>& matrix) {
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

    int n = 10;
  
    vector<vector<double>> matrix(n, vector<double>(n + 1));

    srand(time(0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n + 1; ++j) {
            matrix[i][j] = rand() % 9 + 1; 
        }
    }

    cout << "Исходная матрица:" << endl;
    printMatrix(matrix);
    cout << endl;

    auto start = high_resolution_clock::now();

    vector<double> solution = gaussElimination(matrix);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "Время выполнения: " << duration.count() << " миллисекунд" << endl;

    cout << "Решение СЛАУ:" << endl;
    for (int i = 0; i < n; ++i) {
        cout << "x" << i + 1 << " = " << solution[i] << endl;
    }

    return 0;
}
