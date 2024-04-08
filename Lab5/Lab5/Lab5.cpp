﻿#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

const int N = 4; 
double A[N][N + 1], B[N][N + 1];

int generateRandom() {
    return rand() % 10;
}

void gaussElimination(double A[N][N + 1]) {
	
	// Прямой ход
	for (int k = 0; k < N - 1; k++) {

		for (int i = k + 1; i < N; i++) {
			double koef = A[i][k] / A[k][k];
			for (int j = k; j < N + 1; j++) {
				A[i][j] -= koef * A[k][j];
			}
		}
	}
	// Обратный ход
	for (int k = N - 1; k >= 0; k--) {
		A[k][N] /= A[k][k];
		A[k][k] = 1.0;
		for (int i = 0; i < k; i++) {
			A[i][N] -= A[i][k] * A[k][N];
			A[i][k] = 0.0;
		}
	}
}


int main(int argc, char** argv) {
	int rank;
	int size;
	int block;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	block = N / size;

	if (rank == 0)
	{
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N + 1; ++j) {
				A[i][j] = generateRandom();
				B[i][j] = generateRandom();
			}
		}
	}



	MPI_Bcast(&A[0][0], N * (N + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Прямой ход
	int start = rank * block;
	int end = start + block;
	double temp[N][N + 1];
	double koef = 0;
	for (int k = 0; k < N - 1; k++) {
		
		if (end - 1 > k) {
			for (int i = start; i < end; i++) {
				if (i > k) {
				    koef = A[i][k] / A[k][k];
					for (int j = k; j < N + 1; j++) {
						A[i][j] -= koef * A[k][j];
					}
					
					cout << endl << "iteration = " << k;

					cout << endl << "rank = " << rank << endl;

					cout << "k = " << koef;

					cout << endl;
					for (int i = 0; i < N; ++i) {
						for (int j = 0; j < N; ++j) {
							cout << A[i][j] << "  ";
						}
						cout << endl;
					}
				}
				
			}
		
		}

		MPI_Gather(&A[start][0], block * (N + 1), MPI_DOUBLE, &temp[start][0], block * (N + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

		if (rank == 0)
		{
			for (int i = 0; i < N; i++)
			{
				for (int j = 0; j < N + 1; j++)
				{
					A[i][j] = temp[i][j];
				}
			}
		}
		MPI_Bcast(&A[0][0], N * (N + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

		
	}


	// Обратный ход
	for (int k = N - 1; k >= 0; k--) {
		if (rank == 0)
		{
			A[k][N] /= A[k][k];
			A[k][k] = 1.0;
		}
		MPI_Bcast(&A[k][0], (N + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

		for (int i = start; i < end; i++) {
			if (i < k)
			{
				A[i][N] -= A[i][k] * A[k][N];
				A[i][k] = 0.0;
			}
		}

		MPI_Gather(&A[start][0], block * (N + 1), MPI_DOUBLE, &temp[start][0], block * (N + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

		if (rank == 0)
		{
			for (int i = 0; i < N; i++)
			{
				for (int j = 0; j < N + 1; j++)
				{
					A[i][j] = temp[i][j];
				}
			}
		}

		MPI_Bcast(&A[0][0], N * (N + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

		

		
	}


	if (rank == 0)
	{
		cout << endl;
		for (int i = 0; i < N; ++i) {
			cout << "x[" << i << "] = " << A[i][N] << endl;
		}

		gaussElimination(A);
	}

	MPI_Finalize();
	return 0;
}