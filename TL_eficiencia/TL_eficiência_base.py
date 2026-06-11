import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import factorial
import os

# =============================================================================
# 1. GERAÇÃO DE DADOS SIMULADOS (MOCK DATA)
# =============================================================================
def gerar_dados_teste():
    """
    Gera um conjunto de dados aleatório que mimetiza o comportamento de uma 
    curva de lente térmica (Thermal Lens) para validação do algoritmo.
    """
    t = np.linspace(0, 0.05, 1000) # 50 ms de aquisição
    
    # Parâmetros teóricos arbitrários para a simulação
    theta_real = 0.8
    tc_real = 1.5e-3
    I0_real = 5.0
    m = 1.0
    V = 10e-3 / 1.73e-3 # z / z_op
    
    num = 2 * m * V
    den_a = ((1 + 2 * m)**2 + V**2) / 2
    den_b = 1 + 2 * m + V**2
    
    # Construção do sinal com base no modelo de Shen/Snook
    termo_tempo = tc_real / (t + 1e-12) # Evita divisão por zero
    denominador = den_a * termo_tempo + den_b
    sinal_ideal = I0_real * (1 - (theta_real / 2) * np.arctan(num / denominador))**2
    
    # Adição de ruído Gaussiano branco
    ruido = np.random.normal(0, 0.02, t.shape)
    sinal_ruidoso = sinal_ideal + ruido
    
    return t, sinal_ruidoso

# =============================================================================
# 2. EQUAÇÕES MATEMÁTICAS E AJUSTE (FIT)
# =============================================================================
def calcular_parametros_geometricos(z, z_op, m):
    """
    Calcula os parâmetros de descasamento do feixe (m e V).
    """
    V = z / z_op
    num = 2 * m * V
    den_a = ((1 + 2 * m)**2 + V**2) / 2
    den_b = 1 + 2 * m + V**2
    return num, den_a, den_b

def modelo_lente_termica(t, theta, tc, I0, num, den_a, den_b):
    """
    Equação analítica para a intensidade do feixe de prova no centro (r=0).
    """
    with np.errstate(divide='ignore'):
        termo_tc_t = np.where(t > 0, tc / t, np.inf)
        
    denominador = den_a * termo_tc_t + den_b
    arg_atan = num / denominador
    
    return I0 * (1 - (theta / 2) * np.arctan(arg_atan))**2

def ajustar_sinal_tl(tempo, sinal, num, den_a, den_b):
    """
    Realiza o ajuste não-linear extraindo chutes iniciais diretamente da curva.
    """
    I0 = sinal[0]
    
    # --- Estimativa Dinâmica dos Chutes Iniciais ---
    # Theta aproximado: (I(0) - I(infinito)) / I(0)
    theta_guess = np.abs((I0 - np.mean(sinal[-50:])) / I0)
    
    # tc aproximado: Tempo onde o sinal atinge metade da sua variação total
    meia_variacao = I0 - (I0 - np.mean(sinal[-50:])) / 2
    idx_meio = np.argmin(np.abs(sinal - meia_variacao))
    tc_guess = tempo[idx_meio]
    
    # Proteção caso a estimativa falhe
    if tc_guess <= 0: tc_guess = 1e-3
    if theta_guess <= 0: theta_guess = 0.1
    
    chutes_iniciais = [theta_guess, tc_guess]
    limites = ([-np.inf, 1e-9], [np.inf, np.inf])
    
    funcao_fit = lambda t, theta, tc: modelo_lente_termica(t, theta, tc, I0, num, den_a, den_b)
    
    try:
        popt, _ = curve_fit(funcao_fit, tempo, sinal, p0=chutes_iniciais, bounds=limites)
        theta, tc = popt
        sinal_ajustado = funcao_fit(tempo, theta, tc)
        return theta, tc, sinal_ajustado
    except Exception as e:
        print(f"Erro na convergência do ajuste: {e}")
        return None, None, None

# =============================================================================
# 3. CÁLCULO DE TEMPERATURA E EFICIÊNCIA QUÂNTICA
# =============================================================================
def calcular_delta_T(theta, tc, tempo_array, lambda_probe, L, dndT, n_termos=100):
    """
    Calcula a variação de temperatura máxima (centro do feixe) no estado estacionário.
    """
    if tc <= 0: return np.nan
    t_max = tempo_array[-1]
    
    k = np.arange(1, n_termos + 1)
    termo1 = (-2)**k
    termo2 = k * factorial(k + 1)
    base = 1.0 / (1.0 + 2.0 * t_max / tc)
    termo3 = 1.0 - (base**k)
    
    f_k = (termo1 / termo2) * termo3
    soma_serie = np.sum(f_k)
    termo_log = np.log(1 + 2 * t_max / tc)
    
    deltaT_max = -((np.abs(theta) * lambda_probe) / (4 * np.pi * L * dndT)) * (termo_log + soma_serie)
    return deltaT_max

def calcular_propriedades_quanticas(theta, P_abs, kappa, lambda_p, dndT, lambda_ex, lambda_em):
    """
    Calcula a Eficiência Quântica Não Radiativa (phi) e o Rendimento Quântico (eta).
    """
    phi = - (theta * kappa * lambda_p) / (P_abs * dndT)
    eta = (1 - phi) * (lambda_em / lambda_ex)
    return phi, eta

# =============================================================================
# 4. ROTINA PRINCIPAL E EXPORTAÇÃO GRÁFICA
# =============================================================================
def main():
    # --- Parâmetros Geométricos e do Sistema ---
    z_op = 1.73e-3
    z = 10e-3
    m = 1.0
    
    # --- Parâmetros Físicos e da Amostra (Ex: Rodamina 6G em Etanol) ---
    L = 1e-3
    dndT = -4.0e-4
    kappa = 0.17
    P_abs = 10e-3
    
    # --- Comprimentos de onda (m) ---
    lambda_p = 800e-9
    lambda_ex = 532e-9
    lambda_em = 560e-9

    # 1. Carrega os dados simulados
    tempo, sinal = gerar_dados_teste()
    
    # 2. Calcula parâmetros geométricos
    num, den_a, den_b = calcular_parametros_geometricos(z, z_op, m)
    
    # 3. Realiza o Ajuste
    theta, tc, sinal_ajustado = ajustar_sinal_tl(tempo, sinal, num, den_a, den_b)
    
    if theta is None:
        print("Falha no ajuste. Encerrando.")
        return

    # 4. Cálculos Derivados
    deltaT_max = calcular_delta_T(theta, tc, tempo, lambda_p, L, dndT)
    phi, eta = calcular_propriedades_quanticas(theta, P_abs, kappa, lambda_p, dndT, lambda_ex, lambda_em)
    
    deltaT_curva = deltaT_max * (sinal_ajustado - sinal_ajustado[0]) / (sinal_ajustado[-1] - sinal_ajustado[0])

    # 5. Exibição dos Resultados no Terminal
    print("=== RESULTADOS DO AJUSTE ===")
    print(f"Theta (θ) calculado: {theta:.4f}")
    print(f"Tempo Característico (tc): {tc:.4e} s")
    print(f"Delta T Máximo: {deltaT_max:.4f} °C")
    print("--------------------------------")
    print("=== PROPRIEDADES QUÂNTICAS ===")
    print(f"Eficiência Não Radiativa (φ): {phi:.4f}")
    print(f"Rendimento Quântico (η): {eta:.4f}")

    # 6. Salvamento e EXIBIÇÃO de Gráficos
    os.makedirs("Resultados_TL", exist_ok=True)
    
    # Gráfico 1: Ajuste
    plt.figure(figsize=(8, 5))
    plt.plot(tempo * 1000, sinal, 'k.', alpha=0.3, label="Dados Brutos")
    plt.plot(tempo * 1000, sinal_ajustado, 'r-', linewidth=2, label="Ajuste Shen/Snook")
    plt.xlabel("Tempo (ms)")
    plt.ylabel("Intensidade (V)")
    plt.title(f"Ajuste TL ($\\theta$={theta:.3f}, $t_c$={tc*1000:.2f} ms)")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig("Resultados_TL/Ajuste_Sinal.png", dpi=300)
    plt.show() # <-- Agora o Spyder vai exibir o gráfico na tela

    # Gráfico 2: Temperatura
    plt.figure(figsize=(8, 5))
    plt.plot(tempo * 1000, deltaT_curva, 'b-', linewidth=2, label="Evolução Térmica Ajustada")
    plt.xlabel("Tempo (ms)")
    plt.ylabel(r"$\Delta T$ (°C)")
    plt.title(f"Perfil Temporal de Temperatura (Máx: {deltaT_max:.3f} °C)")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig("Resultados_TL/Evolucao_Temperatura.png", dpi=300)
    plt.show() # <-- Exibe o segundo gráfico
    
    print("\n[Sucesso] Gráficos exibidos na tela e salvos na pasta 'Resultados_TL'.")

if __name__ == "__main__":
    main()