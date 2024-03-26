#include <iostream>
#include <vector>
#include <chrono>
#include <oneapi/tbb.h>

using namespace std;
using namespace std::chrono;

int generateRandom() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> distrib(0, 9);
    return distrib(gen);
}

vector<double> gaussEliminationTBB(const vector<vector<double>>& A, const vector<double>& b) {
    const int n = A.size();
    vector<vector<double>> A_copy = A;
    vector<double> b_copy = b;

    for (int i = 0; i < n; ++i) {
        double pivot = A_copy[i][i];
        if (pivot == 0) {
            throw runtime_error("Matrix is singular");
        }

        tbb::parallel_for(tbb::blocked_range<int>(i + 1, n),
            [&](const tbb::blocked_range<int>& range) {
                for (int j = range.begin(); j < range.end(); ++j) {
                    double ratio = A_copy[j][i] / pivot;
                    for (int k = i; k < n; ++k) {
                        A_copy[j][k] -= ratio * A_copy[i][k];
                    }
                    b_copy[j] -= ratio * b_copy[i];
                }
            });
    }

    vector<double> x(n);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = b_copy[i];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= A_copy[i][j] * x[j];
        }
        x[i] /= A_copy[i][i];
    }

    return x;
}

vector<double> gaussElimination(const vector<vector<double>>& A, const vector<double>& b) {
    const int n = A.size();
    vector<vector<double>> A_copy = A;
    vector<double> b_copy = b;

    for (int i = 0; i < n; ++i) {
        double pivot = A_copy[i][i];
        if (pivot == 0) {
            throw runtime_error("Matrix is singular");
        }

        for (int j = i + 1; j < n; ++j) {
            double ratio = A_copy[j][i] / pivot;
            for (int k = i; k < n; ++k) {
                A_copy[j][k] -= ratio * A_copy[i][k];
            }
            b_copy[j] -= ratio * b_copy[i];
        }
    }

    vector<double> x(n);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = b_copy[i];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= A_copy[i][j] * x[j];
        }
        x[i] /= A_copy[i][i];
    }

    return x;
}

int main() {

    int N = 100; 

    for (int i = 0; i < 100; i++)
    {
        try {
            cout << endl << "N = " << N << endl << endl;

            vector<vector<double>> A(N, vector<double>(N));
            vector<double> b(N);
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    A[i][j] = generateRandom();
                }
                b[i] = generateRandom();
            }

            auto start = high_resolution_clock::now();
            vector<double> x = gaussElimination(A, b);
            auto end = high_resolution_clock::now();
            auto duration_no_tbb = duration_cast<milliseconds>(end - start).count();

            cout << "Time without oneTBB: " << duration_no_tbb << " milliseconds" << endl;

            start = high_resolution_clock::now();
            x = gaussEliminationTBB(A, b);
            end = high_resolution_clock::now();
            auto duration_with_tbb = duration_cast<milliseconds>(end - start).count();

            cout << "Time with oneTBB: " << duration_with_tbb << " milliseconds" << endl;

            N += 100;
        }
        catch(exception)
        {

        }
    }

    return 0;
}
