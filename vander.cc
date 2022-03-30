/*
C++ 1,  JLL 2022.1.5, 3.30
from /home/jinn/OP079C2/selfdrive/modeld/test/polyfit/main.cc

jinn@Liu:~/OP079C2/selfdrive/modeld/test/polyfit$ g++ vander.cc -o vander
jinn@Liu:~/OP079C2/selfdrive/modeld/test/polyfit$ ./vander

References:
Weighted Least Squares (LS) Fit to Data Points (x_i, y_i) by Polynomials (192 in V + 192 in S = 384)
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

Eigen::Matrix<float, MODEL_PATH_DISTANCE, POLYFIT_DEGREE> vander;  // MODEL_PATH_DISTANCE=192, POLYFIT_DEGREE=4 (cubic)

void poly_init(){  // https://nhigham.com/2021/06/15/what-is-a-vandermonde-matrix/
    // Build Vandermonde matrix to fit 192 (x_i, y_i) data points by a cubic (3=4-1) polynomial
  for(int i = 0; i < MODEL_PATH_DISTANCE; i++) {
    for(int j = 0; j < POLYFIT_DEGREE; j++) {
      vander(i, j) = pow(i, POLYFIT_DEGREE-j-1);
      if(i < 10){
        std::cout << "(i, j) = " << i << ", " << j << ", " << "vander(i, j) = " << vander(i, j) << std::endl;
      }
    }
  }
}

void poly_fit(float *in_pts, float *in_stds, float *out) {
    // References to inputs; Eigen::Map is a matrix or vector expression mapping an existing array (in_pts) of data.
    // https://eigen.tuxfamily.org/dox/classEigen_1_1Map.html
  Eigen::Map<Eigen::Matrix<float, MODEL_PATH_DISTANCE, 1> > pts(in_pts, MODEL_PATH_DISTANCE);
  Eigen::Map<Eigen::Matrix<float, MODEL_PATH_DISTANCE, 1> > std(in_stds, MODEL_PATH_DISTANCE);
  Eigen::Map<Eigen::Matrix<float, POLYFIT_DEGREE, 1> > p(out, POLYFIT_DEGREE);

    /* Build Least Squares equations; eigen array()? https://eigen.tuxfamily.org/dox/group__TutorialArrayClass.html
       Array is a class template taking the same template parameters as Matrix.
       If you need to do linear algebraic operations such as matrix multiplication, then you should use matrices;
       if you need to do coefficient-wise operations, then you should use arrays.
       Solve Vandermonde system V/S*P = Y/S for P, input: Y, S, V: 192x4, P: 4x1, Y: 192x1, S: 192x1 (192 in V + 192 in S = 384)
       WVP = WY (W = 1/S: predonditioning or weighting) => weighted LS fit [4] */
  Eigen::Matrix<float, MODEL_PATH_DISTANCE, POLYFIT_DEGREE> lhs = vander.array().colwise() / std.array();  // weighted LS fit [4]
  Eigen::Matrix<float, MODEL_PATH_DISTANCE, 1> rhs = pts.array() / std.array();
    /* std::cout << "sizeof(std.array()) = " << sizeof(std.array()) << std::endl;
    std::cout << "std.rows() x std.cols() = " << std.rows() << " x " << std.cols() << std::endl;
    std::cout << "std.array() = " << std.array() << std::endl;
    std::cout << "std = " << std << std::endl;
    std::cout << "vander.rows() x vander.cols() = " << vander.rows() << " x " << vander.cols() << std::endl;
    std::cout << "vander.array().rowwise().maxCoeff().size() = " << vander.array().rowwise().maxCoeff().size() << std::endl;
    sizeof(std.array()) = 16  // wrong size!
    std.size() = 192
    std.rows() x std.cols() = 192 x 1
    std.array().size() = 192
    vander.rows() x vander.cols() = 192 x 4
    vander.size() = 768
    vander.array().colwise().maxCoeff().size() = 4
    vander.array().rowwise().maxCoeff().size() = 192 */

    // https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
  Eigen::Matrix<float, POLYFIT_DEGREE, 1> scale = 1. / (lhs.array()*lhs.array()).sqrt().colwise().sum();  // L2 norm-like of column vectors
  lhs = lhs * scale.asDiagonal();  // Jacobi-like preconditioning [1], [3]

    // Solve inplace
  Eigen::ColPivHouseholderQR<Eigen::Ref<Eigen::MatrixXf> > qr(lhs);  // QR [4], [5]
  p = qr.solve(rhs);
  p = p.transpose() * scale.asDiagonal();
}

int main(void) {
  poly_init();
  float poly[4];
  poly_fit(pts, stds, poly);
  std::cout << "[" << poly[0] << ", " << poly[1] << ", "
                   << poly[2] << ", " << poly[3] << "]" << std::endl;

  int array[12];  // https://eigen.tuxfamily.org/dox/classEigen_1_1Map.html
  for(int i = 0; i < 12; ++i) array[i] = i;
    // ++i increments the value, then returns it. i++ returns the value, and then increments it.
    // i++ assign -> increment. ++i increment -> assign.
    // For a for loop, use ++i, as it's slightly faster. i++ will create an extra copy that just gets thrown away.
  std::cout << Eigen::Map<Eigen::VectorXi, 1, Eigen::InnerStride<2> > (array, 6)  // the inner stride has already been passed as template parameter
    << std::endl;  // 1 or 0 the same
  std::cout << "C++ version = " << __cplusplus << std::endl;  // 201402 is C++14

  Eigen::MatrixXf a1(2,2);
  Eigen::MatrixXf b1(2,2);
  a1 << 1,2,
        3,4;
  b1 << 5,6,
        7,8;
  std::cout << "a1 * b1 = " << std::endl << a1 * b1 << std::endl;
  Eigen::MatrixXf a(3,2);
  Eigen::MatrixXf b(3,2);
  a << 1,2,
       3,4,
       1,2;
  b << 3,0,
       1,2,
       2,1;
  std::cout << "b.transpose() = " << std::endl << b.transpose() << std::endl;
  std::cout << "a * b.transpose() = " << std::endl << a * b.transpose() << std::endl;
  std::cout << "a.array() * b.array() = " << std::endl << a.array() * b.array() << std::endl;
  auto c = a.array() * b.array();
  std::cout << "c.colwise().sum() = " << std::endl << c.colwise().sum() << std::endl;
}
