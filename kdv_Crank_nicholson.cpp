#include <cmath>
#include <string>
#include <fstream>
#include <iostream>
#include "utils.cpp"
#include <Eigen/SparseCholesky>
#include <Eigen/Sparse>
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include<Eigen/IterativeLinearSolvers>
using Eigen::VectorXd;
// config tá ok
const int space_steps = 2001;                             // número de passos no espaço
const int time_steps = 100000;                            // número de passos no tempo
const double x_init = -100.0;                             // início do intervalo no espaço
const double x_final = 0.0;                               // fim do intervalo no espaço
const double dx = (x_final - x_init) / (space_steps - 1); // incremento espaço
const double dt = 0.0001;                                 // incremento tempo

void general_initial_conditions(VectorXd& ic, double c, double t, double x0) // ta ok
{
    double aux;
    int i;
    for (i = 0; i < space_steps; i++)
    {
        aux = cosh((0.5 * sqrt(c) * (ic[i] - (c * t) - x0)));
        ic[i] = c / (2.0 * aux * aux);
    }
}

void soliton_initial_conditions(VectorXd& ic, int n)
{
    double aux;
    int i;
    for (i = 0; i < space_steps; i++)
    {
        aux = cosh(ic[i]);
        ic[i] = (double)(n) * (n + 1.0) / (aux * aux);
    }
}

void discretize_axis(VectorXd& x) // ta ok
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
double mass_conservation(VectorXd& x)
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
double derivada_diferenca_centrada(VectorXd& valor_variaveis, int indice){
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
void calcula_jacobiano(VectorXd& valor_variaveis_t_mais_1, VectorXd& valor_variaveis_t, Eigen::SparseMatrix<double>& jacobiano){
    double pow_dx_3 = pow(dx,3);
    // m-1 e m-2 fora do dominio, entao nem mechemos nele
    int m = 0;
    jacobiano.coeffRef(m,m) = 1/dt + (1/4*dx)*(valor_variaveis_t_mais_1[m+1] + valor_variaveis_t[m+1] - 0 - 0);
    jacobiano.coeffRef(m,m+1) = (1/4*dx)*(2*valor_variaveis_t_mais_1[m+1]+2*valor_variaveis_t[m+1]+valor_variaveis_t_mais_1[m]+valor_variaveis_t[m]) - (1/(2*pow_dx_3));
    jacobiano.coeffRef(m,m+2) = 1/(4*pow_dx_3); // (u(n,m+2))
    // m-2 fora do dominio, nao acessamos ele
    m = 1;
    jacobiano.coeffRef(m,m-1) = -(1/4*dx)*(2*valor_variaveis_t_mais_1[m-1]+2*valor_variaveis_t[m-1]-valor_variaveis_t_mais_1[m]-valor_variaveis_t[m]) + (1/(2*pow_dx_3));

    jacobiano.coeffRef(m,m) = 1/dt + (1/4*dx)*(valor_variaveis_t_mais_1[m+1] + valor_variaveis_t[m+1] - valor_variaveis_t_mais_1[m-1] - valor_variaveis_t[m-1]);

    jacobiano.coeffRef(m,m+1) = (1/4*dx)*(2*valor_variaveis_t_mais_1[m+1]+2*valor_variaveis_t[m+1]+valor_variaveis_t_mais_1[m]+valor_variaveis_t[m]) - (1/(2*pow_dx_3));

    jacobiano.coeffRef(m,m+2) = 1/(4*pow_dx_3);
    // Vou andando na diagonal...
    for (m = 2; m < space_steps-2; m++){
        jacobiano.coeffRef(m,m-2) = -1/(4*pow_dx_3);
        jacobiano.coeffRef(m,m-1) = -(1/4*dx)*(2*valor_variaveis_t_mais_1[m-1]+2*valor_variaveis_t[m-1]-valor_variaveis_t_mais_1[m]-valor_variaveis_t[m]) + (1/(2*pow_dx_3));
        jacobiano.coeffRef(m,m) = 1/dt + (1/4*dx)*(valor_variaveis_t_mais_1[m+1] + valor_variaveis_t[m+1] - valor_variaveis_t_mais_1[m-1] - valor_variaveis_t[m-1]);
        jacobiano.coeffRef(m,m+1) = (1/4*dx)*(2*valor_variaveis_t_mais_1[m+1]+2*valor_variaveis_t[m+1]+valor_variaveis_t_mais_1[m]+valor_variaveis_t[m]) - (1/(2*pow_dx_3));
        jacobiano.coeffRef(m,m+2) = 1/(4*pow_dx_3);
    }
    // m+2 fora do dominio, nao acessamos ele
    m = space_steps-2;
    jacobiano.coeffRef(m,m-2) = -1/(4*pow_dx_3);
    jacobiano.coeffRef(m,m-1) = -(1/4*dx)*(2*valor_variaveis_t_mais_1[m-1]+2*valor_variaveis_t[m-1]-valor_variaveis_t_mais_1[m]-valor_variaveis_t[m]) + (1/(2*pow_dx_3));
    jacobiano.coeffRef(m,m) = 1/dt + (1/4*dx)*(valor_variaveis_t_mais_1[m+1] + valor_variaveis_t[m+1] - valor_variaveis_t_mais_1[m-1] - valor_variaveis_t[m-1]);
    jacobiano.coeffRef(m,m+1) = (1/4*dx)*(2*valor_variaveis_t_mais_1[m+1]+2*valor_variaveis_t[m+1]+valor_variaveis_t_mais_1[m]+valor_variaveis_t[m]) - (1/(2*pow_dx_3));
    // m+1 e m+2 fora do dominio, entao nem acessamos eles
    m = space_steps-1;
    jacobiano.coeffRef(m,m-2) = -1/(4*pow(dx,3));
    jacobiano.coeffRef(m,m-1) = -(1/4*dx)*(2*valor_variaveis_t_mais_1[m-1]+2*valor_variaveis_t[m-1]-valor_variaveis_t_mais_1[m]-valor_variaveis_t[m]) + (1/(2*pow_dx_3));
    jacobiano.coeffRef(m,m) = 1/dt + (1/4*dx)*(0 + 0 - valor_variaveis_t_mais_1[m-1] - valor_variaveis_t[m-1]);
}
// Calcula F⁰(valor das funcoes que queremos achar as raizes)
//@param valor_variaveis_t_mais_1 chute do valor das raizes que atualiza a cada iteracao
//@param valor_variaveis_t raizes que achamos no tempo anterior
//@param vetor que representa o F⁰
void calcula_funcao_avaliada_chute(VectorXd& valor_variaveis_t_mais_1, VectorXd& valor_variaveis_t, VectorXd& vetor_resultado){
    int m = 0;
    // u_t
    vetor_resultado[m] = ((valor_variaveis_t_mais_1[m]-valor_variaveis_t[m])/dt);
    // u*u_x
    vetor_resultado[m] += (1/(4*dx))*(valor_variaveis_t_mais_1[m+1] + valor_variaveis_t[m+1] + valor_variaveis_t_mais_1[m] + valor_variaveis_t[m] + 0 + 0)
                        * (valor_variaveis_t_mais_1[m+1]+valor_variaveis_t[m+1] - 0 - 0);
    // u_xxx
    vetor_resultado[m] += (1/(4*pow(dx, 3)))*(valor_variaveis_t_mais_1[m+2]+valor_variaveis_t[m+2]-2*(valor_variaveis_t_mais_1[m+1]+valor_variaveis_t[m+1]));
    vetor_resultado[m] *= -1.0;
    m = 1;
    // u_t
    vetor_resultado[m] = ((valor_variaveis_t_mais_1[m]-valor_variaveis_t[m])/dt);
    // u*u_x
    vetor_resultado[m] += (1/(4*dx))*(valor_variaveis_t_mais_1[m+1] + valor_variaveis_t[m+1] + valor_variaveis_t_mais_1[m] + valor_variaveis_t[m]+valor_variaveis_t_mais_1[m-1]+valor_variaveis_t[m-1])
                        * (valor_variaveis_t_mais_1[m+1]+valor_variaveis_t[m+1]-valor_variaveis_t_mais_1[m-1]-valor_variaveis_t[m-1]);
    // u_xxx
    vetor_resultado[m] += (1/(4*pow(dx, 3)))*(valor_variaveis_t_mais_1[m+2]+valor_variaveis_t[m+2]-2*(valor_variaveis_t_mais_1[m+1]+valor_variaveis_t[m+1])+2*(valor_variaveis_t_mais_1[m-1]+valor_variaveis_t[m-1]));
    vetor_resultado[m] *= -1.0;
    for (m = 2; m < space_steps-2; m++){
        vetor_resultado[m] = ((valor_variaveis_t_mais_1[m]-valor_variaveis_t[m])/dt);
        vetor_resultado[m] += (1/(4*dx))*(valor_variaveis_t_mais_1[m+1] + valor_variaveis_t[m+1] + valor_variaveis_t_mais_1[m] + valor_variaveis_t[m]+valor_variaveis_t_mais_1[m-1]+valor_variaveis_t[m-1])
                            * (valor_variaveis_t_mais_1[m+1]+valor_variaveis_t[m+1]-valor_variaveis_t_mais_1[m-1]-valor_variaveis_t[m-1]);
        vetor_resultado[m] += (1/(4*pow(dx, 3)))*(valor_variaveis_t_mais_1[m+2]+valor_variaveis_t[m+2]-2*(valor_variaveis_t_mais_1[m+1]+valor_variaveis_t[m+1])+2*(valor_variaveis_t_mais_1[m-1]+valor_variaveis_t[m-1])-valor_variaveis_t_mais_1[m-2]-valor_variaveis_t[m-2]);
        vetor_resultado[m] *= -1.0;
    }
    m = space_steps-2;
    vetor_resultado[m] = ((valor_variaveis_t_mais_1[m]-valor_variaveis_t[m])/dt);
    vetor_resultado[m] += (1/(4*dx))*(valor_variaveis_t_mais_1[m+1] + valor_variaveis_t[m+1] + valor_variaveis_t_mais_1[m] + valor_variaveis_t[m]+valor_variaveis_t_mais_1[m-1]+valor_variaveis_t[m-1])
                        * (valor_variaveis_t_mais_1[m+1]+valor_variaveis_t[m+1]-valor_variaveis_t_mais_1[m-1]-valor_variaveis_t[m-1]);
    vetor_resultado[m] += (1/(4*pow(dx, 3)))*(-2*(valor_variaveis_t_mais_1[m+1]+valor_variaveis_t[m+1])+2*(valor_variaveis_t_mais_1[m-1]+valor_variaveis_t[m-1])-valor_variaveis_t_mais_1[m-2]-valor_variaveis_t[m-2]);
    vetor_resultado[m] *= -1.0;
    m = space_steps - 1;
    vetor_resultado[m] = ((valor_variaveis_t_mais_1[m]-valor_variaveis_t[m])/dt);
    vetor_resultado[m] += (1/(4*dx))*(valor_variaveis_t_mais_1[m] + valor_variaveis_t[m]+valor_variaveis_t_mais_1[m-1]+valor_variaveis_t[m-1])
                        * (-valor_variaveis_t_mais_1[m-1]-valor_variaveis_t[m-1]);
    vetor_resultado[m] += (1/(4*pow(dx, 3)))*(2*(valor_variaveis_t_mais_1[m-1]+valor_variaveis_t[m-1])-valor_variaveis_t_mais_1[m-2]-valor_variaveis_t[m-2]);
    vetor_resultado[m] *= -1.0;
}
// Rodando o programa
void roda_simulacao_crank_nicholson(){
    // Abrindo arquivos para leitura e escrita
    std::string nome_arquivo_plot = "kdv_data.txt";
    std::string nome_arquivo_condicoes_iniciais = "ic_data.txt";
    std::fstream file_kdv_data(nome_arquivo_plot, std::ios::out);
    std::fstream ic_file(nome_arquivo_condicoes_iniciais, std::ios::out);
    // Criando as condições iniciais...
    VectorXd condicao_inicial(space_steps);
    discretize_axis(condicao_inicial);
    general_initial_conditions(condicao_inicial, 16.0, 0.0, -90.0);
    //Erro será o quão distante de zero as nossas "raizes" estão..
    //Neste caso será o quão distante está a raíz mais distante
    double erro = 500.0;
    VectorXd raizes_tempo_t(space_steps);
    VectorXd raizes_tempo_t_mais_1(space_steps);
    VectorXd vetor_funcao_com_valores_chute(space_steps);
    VectorXd novo_chute(space_steps);
    copia_vetor(condicao_inicial, raizes_tempo_t, space_steps);
    copia_vetor(condicao_inicial, raizes_tempo_t_mais_1, space_steps);
    int count = 0;
    Eigen::SparseMatrix<double> jacobiano(space_steps, space_steps);
    Eigen::IncompleteLUT<double> solver;
    jacobiano.reserve(Eigen::VectorXi::Constant(space_steps, 5));
    for (int j = 0; j < space_steps; j++) ic_file << raizes_tempo_t[j] << std::endl;
    ic_file.close();
    for (int i = 0; i < time_steps; i++){
        // Enquanto o erro é maior que 10⁻⁶
        while (abs(erro) > pow(10, -7) && count < 100){
            // Calcula J e -F⁰
            calcula_jacobiano(raizes_tempo_t_mais_1, raizes_tempo_t, jacobiano);
            calcula_funcao_avaliada_chute(raizes_tempo_t_mais_1, raizes_tempo_t, vetor_funcao_com_valores_chute);
            // Analizamos o padrão da matriz sem se importar com os valores numéricos (apenas na 1 iteracao de todas)
            if (i == 0) solver.analyzePattern(jacobiano);
            // Mudou os valores numericos entao temos que fatorar
            solver.factorize(jacobiano);
            if(solver.info() != Eigen::Success){
                std::cout << "Erro fatoração, possívelmente matriz mal-construída.\n";
                return;
            }
            // Resolvendo o sistema
            novo_chute = solver.solve(vetor_funcao_com_valores_chute) + raizes_tempo_t_mais_1;
            if(solver.info() != Eigen::Success){
                std::cout << "Erro resolver sistema, possívelmente matriz mal-construída.\n";
                return;
            }
            copia_vetor(novo_chute, raizes_tempo_t_mais_1, space_steps);
            erro = valor_max_vetor(vetor_funcao_com_valores_chute, space_steps);
            count++;
        }
        std::cout << count << " iterações do newton para precisao 10⁻⁶\n";
        // Escrevendo no arquivo
        if ((i % 100) == 0){
            for (int j = 0; j < space_steps; j++) file_kdv_data << raizes_tempo_t_mais_1[j] << std::endl;
        }
        // Atualizando o tempo t_n para t_n+1
        copia_vetor(raizes_tempo_t_mais_1, raizes_tempo_t, space_steps);
        erro = 500;
        count = 0;
        std::cout << "tempo: " << i << '\n';
    }
    file_kdv_data.close();
}
// Calculando massa
void cria_arquivo_massa(){
    std::fstream massa_file("massa.txt", std::ios::out);
    double valor_massa = 0.0;
    VectorXd kdv(space_steps);
    std::ifstream kdv_data("kdv_data.txt", std::ios_base::in);
    for (int i = 0; i < (time_steps / 100); i++){
        for (int j = 0; j < space_steps; j++){
            std::string val;
            double val_funcao;
            std::getline(kdv_data, val);
            val_funcao = std::stod(val);
            kdv[j] = val_funcao;
        }
        valor_massa = mass_conservation(kdv);
        massa_file << std::scientific << valor_massa << std::endl;
    }
    
    massa_file.close();
    kdv_data.close();
}
int main(){
    roda_simulacao_crank_nicholson();
    cria_arquivo_massa();
}