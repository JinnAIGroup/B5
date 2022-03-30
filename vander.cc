/*
C++ 1,  JLL 2022.1.5, 3.30
from /home/jinn/OP079C2/selfdrive/modeld/test/polyfit/main.cc

jinn@Liu:~/OP079C2/selfdrive/modeld/test/polyfit$ g++ vander.cc -o vander
jinn@Liu:~/OP079C2/selfdrive/modeld/test/polyfit$ ./vander
*/
#include <iostream>  // What is the difference between #include <filename> and #include "filename"?
#include <cmath>
#include <eigen3/Eigen/Dense>
#include "data.h"

Eigen::Matrix<float, MODEL_PATH_DISTANCE, POLYFIT_DEGREE> vander;

void poly_init(){  // https://nhigham.com/2021/06/15/what-is-a-vandermonde-matrix/
    // Build Vandermonde matrix to fit 192=MODEL_PATH_DISTANCE (x_i, y_i) data points by a polynomial of degree 3=POLYFIT_DEGREE-1
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
       Solve Vandermonde system V/S*P = Y/S for P, input: Y, S, V: 192x4, P: 4x1, Y: 192x1, S: 192x1
       WVP = WY (W = 1/S: predonditioning or weighting) */
  Eigen::Matrix<float, MODEL_PATH_DISTANCE, POLYFIT_DEGREE> lhs = vander.array().colwise() / std.array();
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

  Eigen::Matrix<float, POLYFIT_DEGREE, 1> scale = 1. / (lhs.array()*lhs.array()).sqrt().colwise().sum();
  lhs = lhs * scale.asDiagonal();

    // Solve inplace
  Eigen::ColPivHouseholderQR<Eigen::Ref<Eigen::MatrixXf> > qr(lhs);
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
}
