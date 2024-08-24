#include <cmath>
#include <fstream>
#include <iostream>
#include "utils.cpp"
#include <Eigen/Sparse>
using Eigen::VectorXd;
// config tá ok
const int space_steps = 2001;                             // número de passos no espaço
const int time_steps = 100000;                            // número de passos no tempo
const double x_init = -100.0;                             // início do intervalo no espaço
const double x_final = 0.0;                               // fim do intervalo no espaço
const double dx = (x_final - x_init) / (space_steps - 1); // incremento espaço
const double dt = 0.0001;                                 // incremento tempo

void general_initial_conditions(double *ic, double c, double t, double x0) // ta ok
{
    double aux;
    int i;
    for (i = 0; i < space_steps; i++)
    {
        aux = cosh((0.5 * sqrt(c) * (ic[i] - (c * t) - x0)));
        ic[i] = c / (2.0 * aux * aux);
    }
}

void soliton_initial_conditions(double *ic, int n)
{
    double aux;
    int i;
    for (i = 0; i < space_steps; i++)
    {
        aux = cosh(ic[i]);
        ic[i] = (double)(n) * (n + 1.0) / (aux * aux);
    }
}

void discretize_axis(double *x) // ta ok
{
    int i = 0;
    x[0] = x_init;
    for (i = 1; i < space_steps - 1; i++)
    {
        x[i] = x[i - 1] + dx;
    }
    x[space_steps - 1] = x_final;
    // for (i = 0; i < space_steps; i++)
    // {
    //     std::cout << x[i] << "\n";
    // }
}

// Procedimento que calcula a combinação linear de dois vetores.
// No momento só funciona para esse caso específico, mas pode ser
// facilmente portada.
void linear_combination(double alpha, double *v1, double beta, double *v2, double *combination)
{
    for (int i = 0; i < space_steps; i++)
    {
        combination[i] = (alpha * v1[i]) + (beta * v2[i]);
    }
}
// Norma L2 da função
double mass_conservation(double *x)
{
    // Queremos calcular a raiz quadrada da integral de f(x)², no devido intervalo de integração
    // Usando f(x)² = x[i]² = g(x) e aplicando o método dos trapezios
    // (dx/2) * (g(0) + 2g(1) + ... + 2g(space_steps - 2) + g(space_steps-1)) é o valor da integral aproximado
    // Tiramos a raiz quadrada e retornamos
    double sum = 0.0;
    sum += pow(x[0], 2.0);
    for (int i = 1; i < space_steps - 2; i++)
    {
        sum += 2 * pow(x[i], 2.0);
    }
    sum += pow(x[space_steps - 1], 2);
    sum *= dx / 2;
    return sqrt(sum);
}
// Calcula a derivada aproximada pelas diferenças centradas
//@param valor_variaveis vetor com o valor em cada ponto do espaco
//@param indice ponto em que queremos calcular a derivada
double derivada_diferenca_centrada(double* valor_variaveis, int indice){
    if ((indice-1) < 0){
        return valor_variaveis[indice+1]/(2*dx);
    }
    else if ((indice+1) >= space_steps){
        return -valor_variaveis[indice-1]/(2*dx);
    }
    return (valor_variaveis[indice+1]-valor_variaveis[indice-1])/(2*dx);
}
// Calcula a matriz jacobiana
//@param valor_variaveis_t_mais_1 chute do valor das raizes que atualiza a cada iteracao
//@param valor_variaveis_t raizes que achamos no tempo anterior
//@param jacobiano matriz que representa o jacobiano
void calcula_jacobiano(double* valor_variaveis_t_mais_1, double* valor_variaveis_t, double** jacobiano){
    // m-1 e m-2 fora do dominio, entao nem mechemos nele
    int m = 0;
    jacobiano[m][m] = derivada_diferenca_centrada(valor_variaveis_t_mais_1, m)*((1/dt)-(1/(4*dx))); // (u(n,m))
    jacobiano[m][m+1] = (-1/(4*dx)) * derivada_diferenca_centrada(valor_variaveis_t_mais_1, m+1)*((valor_variaveis_t_mais_1[m+1] + valor_variaveis_t[m+1] - valor_variaveis_t_mais_1[m-1] - valor_variaveis_t[m-1])
                        + (valor_variaveis_t_mais_1[m+1] + valor_variaveis_t[m+1] + valor_variaveis_t_mais_1[m] + valor_variaveis_t[m] + 0.0 + 0.0))
                        + (1/(2*pow(dx, 3)))*(-derivada_diferenca_centrada(valor_variaveis_t_mais_1,m+1)); // (u(n,m+1))
    jacobiano[m][m+2] = derivada_diferenca_centrada(valor_variaveis_t_mais_1,m+2)/(1/(4*pow(dx,3))); // (u(n,m+2))
    // m-2 fora do dominio, nao acessamos ele
    m = 1;
    jacobiano[m][m-1] = (-1/(4*dx))*(derivada_diferenca_centrada(valor_variaveis_t_mais_1,m-1))*(valor_variaveis_t_mais_1[m+1]+valor_variaveis_t[m+1]-valor_variaveis_t_mais_1[m-1]-valor_variaveis_t[m-1]
    -(valor_variaveis_t_mais_1[m+1]+valor_variaveis_t[m+1]+valor_variaveis_t_mais_1[m]+valor_variaveis_t[m]+valor_variaveis_t_mais_1[m-1]+valor_variaveis_t[m-1]));
    jacobiano[m][m] = derivada_diferenca_centrada(valor_variaveis_t_mais_1, m)*((1/dt)-(1/(4*dx)));
    jacobiano[m][m+1] = (-1/(4*dx)) * derivada_diferenca_centrada(valor_variaveis_t_mais_1, m+1)*((valor_variaveis_t_mais_1[m+1] + valor_variaveis_t[m+1] - valor_variaveis_t_mais_1[m-1] - valor_variaveis_t[m-1])
                        + (valor_variaveis_t_mais_1[m+1] + valor_variaveis_t[m+1] + valor_variaveis_t_mais_1[m] + valor_variaveis_t[m] + valor_variaveis_t_mais_1[m-1] + valor_variaveis_t[m-1]))
                        + (1/(2*pow(dx, 3)))*(-derivada_diferenca_centrada(valor_variaveis_t_mais_1,m+1));
    jacobiano[m][m+2] = derivada_diferenca_centrada(valor_variaveis_t_mais_1,m+2)/(1/(4*pow(dx,3)));
    // Vou andando na diagonal...
    for (m = 2; m < space_steps-2; m++){
        jacobiano[m][m-2] = derivada_diferenca_centrada(valor_variaveis_t_mais_1, m-2)/(4*pow(dx,3));
        jacobiano[m][m-1] = (-1/(4*dx))*(derivada_diferenca_centrada(valor_variaveis_t_mais_1,m-1))*(valor_variaveis_t_mais_1[m+1]+valor_variaveis_t[m+1]-valor_variaveis_t_mais_1[m-1]-valor_variaveis_t[m-1]
    -(valor_variaveis_t_mais_1[m+1]+valor_variaveis_t[m+1]+valor_variaveis_t_mais_1[m]+valor_variaveis_t[m]+valor_variaveis_t_mais_1[m-1]+valor_variaveis_t[m-1]));;
        jacobiano[m][m] = derivada_diferenca_centrada(valor_variaveis_t_mais_1, m)*((1/dt)-(1/(4*dx)));
        jacobiano[m][m+1] = (-1/(4*dx)) * derivada_diferenca_centrada(valor_variaveis_t_mais_1, m+1)*((valor_variaveis_t_mais_1[m+1] + valor_variaveis_t[m+1] - valor_variaveis_t_mais_1[m-1] - valor_variaveis_t[m-1])
                        + (valor_variaveis_t_mais_1[m+1] + valor_variaveis_t[m+1] + valor_variaveis_t_mais_1[m] + valor_variaveis_t[m] + valor_variaveis_t_mais_1[m-1] + valor_variaveis_t[m-1]))
                        + (1/(2*pow(dx, 3)))*(-derivada_diferenca_centrada(valor_variaveis_t_mais_1,m+1));
        jacobiano[m][m+2] = derivada_diferenca_centrada(valor_variaveis_t_mais_1,m+2)/(1/(4*pow(dx,3)));
    }
    // m+2 fora do dominio, nao acessamos ele
    m = space_steps-2;
    jacobiano[m][m-2] = derivada_diferenca_centrada(valor_variaveis_t_mais_1, m-2)/(4*pow(dx,3));
    jacobiano[m][m-1] = (-1/(4*dx))*(derivada_diferenca_centrada(valor_variaveis_t_mais_1,m-1))*(valor_variaveis_t_mais_1[m+1]+valor_variaveis_t[m+1]-valor_variaveis_t_mais_1[m-1]-valor_variaveis_t[m-1]
    -(valor_variaveis_t_mais_1[m+1]+valor_variaveis_t[m+1]+valor_variaveis_t_mais_1[m]+valor_variaveis_t[m]+valor_variaveis_t_mais_1[m-1]+valor_variaveis_t[m-1]));;
    jacobiano[m][m] = derivada_diferenca_centrada(valor_variaveis_t_mais_1, m)*((1/dt)-(1/(4*dx)));
    jacobiano[m][m+1] = (-1/(4*dx)) * derivada_diferenca_centrada(valor_variaveis_t_mais_1, m+1)*((valor_variaveis_t_mais_1[m+1] + valor_variaveis_t[m+1] - valor_variaveis_t_mais_1[m-1] - valor_variaveis_t[m-1])
                    + (valor_variaveis_t_mais_1[m+1] + valor_variaveis_t[m+1] + valor_variaveis_t_mais_1[m] + valor_variaveis_t[m] + valor_variaveis_t_mais_1[m-1] + valor_variaveis_t[m-1]))
                    + (1/(2*pow(dx, 3)))*(-derivada_diferenca_centrada(valor_variaveis_t_mais_1,m+1));
    // m+1 e m+2 fora do dominio, entao nem acessamos eles
    m = space_steps-1;
    jacobiano[m][m-2] = derivada_diferenca_centrada(valor_variaveis_t_mais_1, m-2)/(4*pow(dx,3));
    jacobiano[m][m-1] = (-1/(4*dx))*(derivada_diferenca_centrada(valor_variaveis_t_mais_1,m-1))*(valor_variaveis_t_mais_1[m+1]+valor_variaveis_t[m+1]-valor_variaveis_t_mais_1[m-1]-valor_variaveis_t[m-1]
    -(valor_variaveis_t_mais_1[m+1]+valor_variaveis_t[m+1]+valor_variaveis_t_mais_1[m]+valor_variaveis_t[m]+valor_variaveis_t_mais_1[m-1]+valor_variaveis_t[m-1]));
    jacobiano[m][m] = derivada_diferenca_centrada(valor_variaveis_t_mais_1, m)*((1/dt)-(1/(4*dx)));
}
// Calcula F⁰(valor das funcoes que queremos achar as raizes)
//@param valor_variaveis_t_mais_1 chute do valor das raizes que atualiza a cada iteracao
//@param valor_variaveis_t raizes que achamos no tempo anterior
//@param vetor que representa o F⁰
double* resolve_sistema(double** A, double* X, double* B){

}
void calcula_funcao_avaliada_chute(double* valor_variaveis_t_mais_1, double* valor_variaveis_t, double* vetor_resultado){
    int m = 0;
    vetor_resultado[m] = ((valor_variaveis_t_mais_1[m]-valor_variaveis_t[m])/dt);
    vetor_resultado[m] += (1/(4*dx))*(valor_variaveis_t_mais_1[m+1] + valor_variaveis_t[m+1] + valor_variaveis_t_mais_1[m] + valor_variaveis_t[m])
                        * (valor_variaveis_t_mais_1[m+1]+valor_variaveis_t[m+1]);
    vetor_resultado[m] += (1/(4*pow(dx, 3)))*(valor_variaveis_t_mais_1[m+2]+valor_variaveis_t[m+2]-2*(valor_variaveis_t_mais_1[m+1]+valor_variaveis_t[m+1]));
    m = 1;
    vetor_resultado[m] = ((valor_variaveis_t_mais_1[m]-valor_variaveis_t[m])/dt);
    vetor_resultado[m] += (1/(4*dx))*(valor_variaveis_t_mais_1[m+1] + valor_variaveis_t[m+1] + valor_variaveis_t_mais_1[m] + valor_variaveis_t[m]+valor_variaveis_t_mais_1[m-1]+valor_variaveis_t[m-1])
                        * (valor_variaveis_t_mais_1[m+1]+valor_variaveis_t[m+1]-valor_variaveis_t_mais_1[m-1]-valor_variaveis_t[m-1]);
    vetor_resultado[m] += (1/(4*pow(dx, 3)))*(valor_variaveis_t_mais_1[m+2]+valor_variaveis_t[m+2]-2*(valor_variaveis_t_mais_1[m+1]+valor_variaveis_t[m+1])+2*(valor_variaveis_t_mais_1[m-1]+valor_variaveis_t[m-1]));
    for (m = 2; m < space_steps-2; m++){
        vetor_resultado[m] = ((valor_variaveis_t_mais_1[m]-valor_variaveis_t[m])/dt);
        vetor_resultado[m] += (1/(4*dx))*(valor_variaveis_t_mais_1[m+1] + valor_variaveis_t[m+1] + valor_variaveis_t_mais_1[m] + valor_variaveis_t[m]+valor_variaveis_t_mais_1[m-1]+valor_variaveis_t[m-1])
                            * (valor_variaveis_t_mais_1[m+1]+valor_variaveis_t[m+1]-valor_variaveis_t_mais_1[m-1]-valor_variaveis_t[m-1]);
        vetor_resultado[m] += (1/(4*pow(dx, 3)))*(valor_variaveis_t_mais_1[m+2]+valor_variaveis_t[m+2]-2*(valor_variaveis_t_mais_1[m+1]+valor_variaveis_t[m+1])+2*(valor_variaveis_t_mais_1[m-1]+valor_variaveis_t[m-1])-valor_variaveis_t_mais_1[m-2]-valor_variaveis_t[m-2]);
    }
    m = space_steps-2;
    vetor_resultado[m] = ((valor_variaveis_t_mais_1[m]-valor_variaveis_t[m])/dt);
    vetor_resultado[m] += (1/(4*dx))*(valor_variaveis_t_mais_1[m+1] + valor_variaveis_t[m+1] + valor_variaveis_t_mais_1[m] + valor_variaveis_t[m]+valor_variaveis_t_mais_1[m-1]+valor_variaveis_t[m-1])
                        * (valor_variaveis_t_mais_1[m+1]+valor_variaveis_t[m+1]-valor_variaveis_t_mais_1[m-1]-valor_variaveis_t[m-1]);
    vetor_resultado[m] += (1/(4*pow(dx, 3)))*(-2*(valor_variaveis_t_mais_1[m+1]+valor_variaveis_t[m+1])+2*(valor_variaveis_t_mais_1[m-1]+valor_variaveis_t[m-1])-valor_variaveis_t_mais_1[m-2]-valor_variaveis_t[m-2]);
    m = space_steps - 1;
    vetor_resultado[m] = ((valor_variaveis_t_mais_1[m]-valor_variaveis_t[m])/dt);
    vetor_resultado[m] += (1/(4*dx))*(valor_variaveis_t_mais_1[m] + valor_variaveis_t[m]+valor_variaveis_t_mais_1[m-1]+valor_variaveis_t[m-1])
                        * (-valor_variaveis_t_mais_1[m-1]-valor_variaveis_t[m-1]);
    vetor_resultado[m] += (1/(4*pow(dx, 3)))*(2*(valor_variaveis_t_mais_1[m-1]+valor_variaveis_t[m-1])-valor_variaveis_t_mais_1[m-2]-valor_variaveis_t[m-2]);
}
int main(){
    // Criando as condições iniciais...
    double* chute_inicial = new double[space_steps];
    discretize_axis(chute_inicial);
    general_initial_conditions(chute_inicial, 16.0, 0.0, -90.0);
    // Erro será o quão distante de zero as nossas "raizes" estão..
    // Neste caso será o quão distante está a raíz mais distante
    double erro = 500.0;
    double* raizes_tempo_anterior = new double[space_steps];
    double* raizes_tempo_atual = new double[space_steps];
    copia_vetor(chute_inicial, raizes_tempo_anterior, space_steps);
    copia_vetor(chute_inicial, raizes_tempo_atual, space_steps);
    int count = 0;
    for (int i = 0; i < time_steps; i++){
        // Enquanto o erro é maior que 10⁻³
        while (erro > 0.001 && count < 100){
            // Alocando jacobiano
            double** jacobiano = new double*[space_steps];
            for (int i = 0; i < space_steps; i++) jacobiano[i] = new double[space_steps];
            double* vet_funcao_avaliada_chute = new double[space_steps];
            // Calculo F⁰
            calcula_funcao_avaliada_chute(raizes_tempo_atual,raizes_tempo_anterior,vet_funcao_avaliada_chute);
            calcula_jacobiano(raizes_tempo_atual, raizes_tempo_anterior, jacobiano);
            for (int i = 0; i < space_steps; i++) vet_funcao_avaliada_chute[i] *= -1;
            double* novo_chute_raiz = new double[space_steps];
            resolve_sistema(jacobiano, novo_chute_raiz, vet_funcao_avaliada_chute);
            // dou free nas raizes_tempo_anterior ja que nao vou usar mais
            delete[] raizes_tempo_anterior;
            // Atualizo as raizes no tempo t e t-1
            raizes_tempo_anterior = raizes_tempo_atual;
            raizes_tempo_atual = novo_chute_raiz;
            for (int i = 0; i < space_steps; i++) delete[] jacobiano[i];
            delete[] jacobiano;
            delete[] vet_funcao_avaliada_chute;
            count++;
        }
        count = 0;
    }
}
