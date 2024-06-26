#define DLLEXPORT extern "C" __declspec(dllexport)

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

using namespace std;
using namespace Eigen;

DLLEXPORT void solve_principal_component_analysis(double* m_array_input,
                                                  const int rows,
                                                  const int columns,
                                                  double* v_array_mean_out,
                                                  double* m_array_cov_out,
                                                  double* v_array_eigenvalues_out,
                                                  double* m_array_eigenvectors_out)
{
    cout << "--- C++ codes (START) ---" << endl;
    cout << "--- solve_principal_component_analysis (START) ---" << endl;
  
    // Map m_array_input to MatrixXd
    // rows: Number of sample row vectors
    // columns: Size of sample row vectors
    // - Make memory array Row Major to map m_array_input (Eigen's default order is Column Major)
    // - C# two dimensional array double[,] is Row Major
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m_input(m_array_input, rows, columns);

    cout << "m_input is prepared." << endl;

    // ---------------------------------------------------------------
    // Calculation of Covariance Matrix
    // ---------------------------------------------------------------

    // Map v_array_mean_out to VectorXd
    // - The size of mean vector is columns
    Map<VectorXd> v_mean(v_array_mean_out, columns);

    // Calculate the mean of sample row vectors
    v_mean = m_input.colwise().mean();

    cout << "v_mean" << endl;
    cout << v_mean(0) << ", ..., " << v_mean(columns - 1) << endl << endl;    
    
    // Subtract v_mean.transpose() from each sample row vector
    m_input = m_input.rowwise() - v_mean.transpose();

    // Map m_array_cov_out to MatrixXd
    // - The size of rows and columns are the same as the input sample row vectors
    // - Make memory array Row Major to map m_array_cov_out
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m_cov(m_array_cov_out, columns, columns);

    // Calculate covariance matrix
    // - Ignore the effect of constant multiplication
    // - Skip division by N or N - 1 (N is sample size)
    m_cov = m_input.adjoint() * m_input;

    cout << "m_cov is prepared." << endl;
    cout << m_cov(0, 0) << ", ..., " << m_cov(0, columns - 1) << endl;
    cout << "..." << endl;    
    cout << m_cov(columns - 1, 0) << ", ..., " << m_cov(columns - 1, columns - 1) << endl << endl;
    
    // ---------------------------------------------------------------
    // Calculate Eigendecomposition of a covariance matrix
    // ---------------------------------------------------------------
    SelfAdjointEigenSolver<MatrixXd> esolver(m_cov);

    // Map v_array_eigenvalues_out to VectorXd
    Map<VectorXd> eigenvalues(v_array_eigenvalues_out, columns);
    // Apply reverse() to make the order of eigenvalues decreasing order
    eigenvalues = esolver.eigenvalues().reverse().eval();

    cout << "eigenvalues" << endl;
    cout << eigenvalues(0) << ", ..., " << eigenvalues(columns - 1) << endl << endl;
    
    // Map m_array_eigenvectors_out to MatrixXd
    // - Make memory array Row Major to map m_array_cov_out
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> eigenvectors(m_array_eigenvectors_out, columns, columns);

    // Apply reverse() to eigenvectors because the order of eigenvalues are reversed
    eigenvectors = esolver.eigenvectors().rowwise().reverse().eval();

    cout << "eigenvectors" << endl;
    cout << eigenvectors(0, 0) << ", ..., " << eigenvectors(0, columns - 1) << endl;
    cout << "..." << endl;    
    cout << eigenvectors(columns - 1, 0) << ", ..., " << eigenvectors(columns - 1, columns - 1) << endl << endl;

    cout << "Check Result: P L P^t" << endl;
    MatrixXd PLPt = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
    cout << PLPt(0, 0) << ", ..., " << PLPt(0, columns - 1) << endl;
    cout << "..." << endl;    
    cout << PLPt(columns - 1, 0) << ", ..., " << PLPt(columns - 1, columns - 1) << endl << endl;

    cout << "--- solve_principal_component_analysis (END) ---" << endl;
    cout << "--- C++ codes (END) ---" << endl;
}
