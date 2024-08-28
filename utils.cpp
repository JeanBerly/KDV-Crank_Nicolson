#include <Eigen/StdVector>
using Eigen::VectorXd;
void copia_vetor(VectorXd& origem, VectorXd& destino, long tamanho){
    for (long i = 0; i < tamanho; i++){
        destino[i] = origem[i];
    }
}
double valor_max_vetor(VectorXd& vec, int tam){
    double max = 0;
    for (int i = 0; i < tam; i++){
        if (abs(vec[i]) > max) max = abs(vec[i]);
    }
    return max;
}