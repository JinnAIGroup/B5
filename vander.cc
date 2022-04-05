/*
C++ 1,  JLL 2022.1.5, 4.5
from /home/jinn/OP079C2/selfdrive/modeld/test/polyfit/main.cc

jinn@Liu:~/OP079C2/selfdrive/modeld/test/polyfit$ g++ vander.cc -o vander
jinn@Liu:~/OP079C2/selfdrive/modeld/test/polyfit$ ./vander

References:
Weighted Least Squares (LS) Fit to 192+192 Data Points (x_i,y_i)+(x_i,s_i) by a Cubic Polynomial (192 x_i in V(192,4) + 192 s_i in S(192,1) = 384)
[1] Liu, J.-L. Lecture 6 Symmetric SOR (SSOR); http://www.nhcue.edu.tw/~jinnliu/teaching/LiuLec17x/LiuLec17x.htm
[2] Liu, J.-L. Lecture 7 Conjugate Gradient Method (CG)
[3] France, A. C. (2004). Condition number of Vandermonde matrix in least-squares polynomial fitting problems.
[4] Saraswat, J. (2009). A study of Vandermonde-like matrix systems with emphasis on preconditioning and Krylov matrix connection (Doctoral dissertation, University of Kansas).
[5] Reichel, L. (1991). Fast QR Decomposition of Vandermonde-Like Mmatrices and Polynomial Least Squares Approximation. SIAM journal on matrix analysis and applications, 12(3), 552-564.
*/
#include <iostream>  // What is the difference between #include <filename> and #include "filename"?
#include <cmath>
#include <eigen3/Eigen/Dense>
#include "data.h"

Eigen::Matrix<float, MODEL_PATH_DISTANCE, POLYFIT_DEGREE> vander;  // 192x4 matrix; MODEL_PATH_DISTANCE=192, POLYFIT_DEGREE=4 (cubic)

void poly_init(){  // https://nhigham.com/2021/06/15/what-is-a-vandermonde-matrix/
    // Build Vandermonde matrix to fit 192 (x_i,y_i) data points by a cubic (3=4-1) polynomial
  for(int i = 0; i < MODEL_PATH_DISTANCE; i++) {
    for(int j = 0; j < POLYFIT_DEGREE; j++) {
      vander(i, j) = pow(i, POLYFIT_DEGREE-j-1);  // x_i = 0, ..., 191 => V(i,j)
      if(i < 10){
        std::cout << "(i, j) = " << i << ", " << j << ", " << "vander(i, j) = " << vander(i, j) << std::endl;
      }
    }
  }
}

void poly_fit(float *in_pts, float *in_stds, float *out) {
    // References to inputs; Eigen::Map maps an existing array (in_pts) of data to a matrix or vector (pts).
    // https://eigen.tuxfamily.org/dox/classEigen_1_1Map.html
  Eigen::Map<Eigen::Matrix<float, MODEL_PATH_DISTANCE, 1> > pts(in_pts, MODEL_PATH_DISTANCE);  // 192x1 vector; y_i = pts_i in data.h
  Eigen::Map<Eigen::Matrix<float, MODEL_PATH_DISTANCE, 1> > std(in_stds, MODEL_PATH_DISTANCE);  // 192x1 vector; s_i = std_i in data.h
  Eigen::Map<Eigen::Matrix<float, POLYFIT_DEGREE, 1> > p(out, POLYFIT_DEGREE);  // 4x1 matrix; output of vander.cc: p_j, j = 0, ..., 3, 4 coefficeints of the cubic polynomial

    /* Build Least Squares equations; eigen array()? https://eigen.tuxfamily.org/dox/group__TutorialArrayClass.html
       Array is a class template taking the same template parameters as Matrix.
       If you need to do linear algebraic operations such as matrix multiplication, then you should use matrices;
       if you need to do coefficient-wise operations, then you should use arrays.
       Solve Vandermonde system V/S*P = Y/S for P = <p_j>; input: Y, S; V(i,j): 192x4, P: 4x1, Y(i,1)=<y_i>: 192x1, S(i,1)=<s_i>: 192x1
       WVP = WY (W = 1/S: predonditioning or weighting) => weighted LS fit [4] */
  Eigen::Matrix<float, MODEL_PATH_DISTANCE, POLYFIT_DEGREE> lhs = vander.array().colwise() / std.array();  // 192x4 matrix; weighted LS fit [4]
  Eigen::Matrix<float, MODEL_PATH_DISTANCE, 1> rhs = pts.array() / std.array();  // 192x1 vector;
    /* std::cout << "sizeof(std.array()) = " << sizeof(std.array()) << std::endl;
      sizeof(std.array()) = 16  // wrong size!
    std::cout << "std = " << std << std::endl;
      std.size() = 192
    std::cout << "std.rows() x std.cols() = " << std.rows() << " x " << std.cols() << std::endl;
      std.rows() x std.cols() = 192x1
    std::cout << "std.array() = " << std.array() << std::endl;
      std.array().size() = 192
    std::cout << "vander.rows() x vander.cols() = " << vander.rows() << " x " << vander.cols() << std::endl;
      vander.rows() x vander.cols() = 192x4
    std::cout << "vander.array().rowwise().maxCoeff().size() = " << vander.array().rowwise().maxCoeff().size() << std::endl;
      vander.size() = 768
      vander.array().colwise().maxCoeff().size() = 4
      vander.array().rowwise().maxCoeff().size() = 192 */

    // https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
  Eigen::Matrix<float, POLYFIT_DEGREE, 1> scale = 1. / (lhs.array()*lhs.array()).sqrt().colwise().sum();  // Normalization, L2 norm-like of column vectors
  lhs = lhs * scale.asDiagonal();  // Jacobi-like preconditioning [1], [3]

    // Solve inplace (Ref<>)
  Eigen::ColPivHouseholderQR<Eigen::Ref<Eigen::MatrixXf> > qr(lhs);  // QR [4], [5]
  p = qr.solve(rhs);  // https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
  p = p.transpose() * scale.asDiagonal();
}

int main(void) {
  std::cout << "C++ version = " << __cplusplus << std::endl;  // 201402 is C++14
  poly_init();  // create 192x4 vander matrix
  float poly[4];
  poly_fit(pts, stds, poly);  // solve vander (weighted LS) system for the coefficient poly[4]
  std::cout << "[" << poly[0] << ", " << poly[1] << ", "
                   << poly[2] << ", " << poly[3] << "]" << std::endl;

  int array[12];  // https://eigen.tuxfamily.org/dox/classEigen_1_1Map.html
  for(int i = 0; i < 12; ++i) array[i] = i;
    // ++i increments the value, then returns it. i++ returns the value, and then increments it.
    // i++ assign -> increment. ++i increment -> assign.
    // For a for loop, use ++i, as it's slightly faster. i++ will create an extra copy that just gets thrown away.
  std::cout << Eigen::Map<Eigen::VectorXi, 1, Eigen::InnerStride<2> > (array, 6)  // maps array to matrix; the inner stride has already been passed as template parameter
    << std::endl;  // 1 or 0 the same

  Eigen::MatrixXf A(3,2);
  Eigen::MatrixXf B(3,2);
  A << 1,2,
       3,4,
       1,2;
  B << 3,0,
       1,2,
       2,1;
  std::cout << "B.transpose() = " << std::endl << B.transpose() << std::endl;
  std::cout << "A * B.transpose() = " << std::endl << A * B.transpose() << std::endl;
  std::cout << "A.array() * B.array() = " << std::endl << A.array() * B.array() << std::endl;  // matrix to array
  auto C = A.array() * B.array();
  std::cout << "C.colwise().sum() = " << std::endl << C.colwise().sum() << std::endl;

  Eigen::MatrixXf A1(2,2);
  Eigen::MatrixXf B1(2,2);
  A1 << 1,2,
        3,4;
  B1 << 5,6,
        7,8;
  std::cout << "A1 * B1 = " << std::endl << A1 * B1 << std::endl;

    // Inplace matrix decompositions, https://eigen.tuxfamily.org/dox/group__InplaceDecomposition.html
  Eigen::MatrixXf A2(2,2); A2 << 2,-1,1,3;
  auto A0 = A2; // save A2
  std::cout << "A2 before decomposition:\n" << A2 << std::endl;
  Eigen::PartialPivLU<Eigen::Ref<Eigen::MatrixXf> > lu(A2);
    // LU, Cholesky, and QR decompositions can operate inplace with a Ref<>, i.e.,
    // lu object computes and stores the L and U factors within the memory held by the matrix A2, i.e., A2 is lost.
  std::cout << "A2 after decomposition:\n" << A2 << std::endl;
  Eigen::VectorXf b(2); b << 1,4;
  Eigen::VectorXf x = lu.solve(b);
  std::cout << "x = " << std::endl << x << std::endl;
  std::cout << "Residual: " << (A0 * x - b).norm() << std::endl;

  Eigen::MatrixXf A3(2,2);
  A3 << 3,-2,3,1;
  lu.compute(A3);  // A2 in lu(A2) is changed again
  std::cout << "the matrix storing the L and U factors:\n" << lu.matrixLU() << std::endl;
  std::cout << "A2 after decomposition:\n" << A2 << std::endl;
  std::cout << "A3 is unchangted in lu.compute(A3):\n" << A3 << std::endl;
  Eigen::VectorXf x1 = lu.solve(b);
  std::cout << "x1 = " << std::endl << x1 << std::endl;
  std::cout << "Residual: " << (A3 * x1 - b).norm() << std::endl;
}
