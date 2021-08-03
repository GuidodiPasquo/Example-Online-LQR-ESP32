#include "esp32-hal-cpu.h"

#include <Eigen.h>
#include <iostream>
#include <Dense>
#include <vector>

using namespace Eigen;
using namespace std;

MatrixXd P_1(3, 3), P(3, 3), A(3, 3), A_tr(3, 3), Q(3, 3), R(1, 1), B(3, 1), B_tr(1, 3), aux11(1, 1), APA(3, 3), APB(3, 1), BPA(1, 3), R_BPBi(1, 1), K(1, 3);

MatrixXd H(6, 6), Ac(3, 3), Bc(3, 1), aux11c(1, 1), Rc(1, 1), Qc(3, 3), BRB(3, 3);

void setup() {
  Serial.begin(115200);
  setCpuFrequencyMhz(240);

  //////////////////////////////Eigen value method
  Ac << 0, 1.0, 0,
  0, 0, 2 * 8.28404538,
  -0.49, 1.0, -2 * 0.15999644;

  Bc << 0,
  20.02930514 / 2,
  0.49;

  aux11c = Rc.inverse();
  BRB = ((Bc * aux11c) * Bc.transpose());

  H << Ac,     -BRB,
  -Qc, -(Ac.transpose());

  cout << "H =" << endl << H << endl;

  MatrixXcf  Hcomplex;
  timer = micros();
  es.compute(H);
  timer = micros() - timer;
  Serial.print(timer);
  cout << "val =" << endl << es.eigenvalues() << endl;
  cout << "vec =" << endl << es.eigenvectors() << endl;

  timer = micros();
  P = solveRiccati(A, B, Q, R, P);
  //K=Rc.inverse()*(Bc.transpose()*P);
  K = (aux11.inverse()) * (B_tr * (P * A));
  K = -K;
  timer = micros() - timer;
  Serial.print(timer);

  cout << "vec =" << endl << P << endl;
  cout << "vec =" << endl << K << endl;
}

void loop()
{
}




//Thanks Taka Horibe -> https://github.com/TakaHoribe/Riccati_Solver

MatrixXd solveRiccati(const Eigen::MatrixXd &A,
                      const Eigen::MatrixXd &B,
                      const Eigen::MatrixXd &Q,
                      const Eigen::MatrixXd &R, Eigen::MatrixXd &P) {

  const uint dim_x = A.rows();
  const uint dim_u = B.cols();

  // set Hamilton matrix
  Eigen::MatrixXd Ham = Eigen::MatrixXd::Zero(2 * dim_x, 2 * dim_x);
  Ham << A, -B * R.inverse() * B.transpose(), -Q, -A.transpose();

  // calc eigenvalues and eigenvectors
  Eigen::EigenSolver<Eigen::MatrixXd> Eigs(Ham);

  // check eigen values
  // std::cout << "eigen values：\n" << Eigs.eigenvalues() << std::endl;
  // std::cout << "eigen vectors：\n" << Eigs.eigenvectors() << std::endl;

  // extract stable eigenvectors into 'eigvec'
  Eigen::MatrixXcd eigvec = Eigen::MatrixXcd::Zero(2 * dim_x, dim_x);
  int j = 0;
  for (int i = 0; i < 2 * dim_x; ++i) {
    if (Eigs.eigenvalues()[i].real() < 0.) {
      eigvec.col(j) = Eigs.eigenvectors().block(0, i, 2 * dim_x, 1);
      ++j;
    }
  }

  // calc P with stable eigen vector matrix
  Eigen::MatrixXcd Vs_1, Vs_2;
  Vs_1 = eigvec.block(0, 0, dim_x, dim_x);
  Vs_2 = eigvec.block(dim_x, 0, dim_x, dim_x);
  P = (Vs_2 * Vs_1.inverse()).real();
  return P;
}
