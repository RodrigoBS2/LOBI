# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 16:04:16 2026

@author: Flávia Eduarda
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog

# =============================================================================
# 1. PARÂMETROS FÍSICOS CONSTANTES 
# =============================================================================

# -- Parâmetros Geométricos (Montagem Óptica) --
Z_OP = 1.9e-3       # Z_OP (m): Posição confocal da lente. É a distância em que o feixe de laser dobra de área.
Z = 1.05e-2          # Z (m): Distância da amostra (cubeta) até o ponto focal (cintura) do laser de excitação.
L = 1e-3            # L (m): Espessura da amostra (caminho óptico dentro da cubeta).
M_RATIO = 1.0       # m: Grau de descasamento (mismatch) entre o feixe de excitação e o feixe de prova. 
PERDA_CUBETA = 0.10 # Perda de Fresnel (%): Fração da luz perdida por reflexão no vidro antes de entrar no líquido (10%).

# -- Propriedades do Solvente --
DNDT = -3.94e-4     # dn/dT (K^-1): Como o índice de refração muda ao esquentar. Negativo significa que forma lente divergente.
KAPPA = 0.171       # kappa (W/m.K): Condutividade térmica. Define quão rápido o calor se espalha.

# -- Propriedades da Amostra e Laser --
P_IN = 17e-3        # P_in (W): Potência bruta do laser de excitação que sai do equipamento.
A_ABS = 0.3567      # A: Absorbância da amostra (Rodamina 6G) no comprimento de onda do laser de excitação.
LAMBDA_EX = 532e-9  # Lambda_ex (m): Comprimento de onda do laser que excita a amostra.
LAMBDA_EM = 560e-9  # Lambda_em (m): Comprimento de onda central da luz emitida por fluorescência.

# -- Limites Matemáticos do Ajuste (Evita que o programa faça chutes absurdos) --
TC_MIN = 1e-4       # Tempo característico mínimo permitido
TC_MAX = 50e-3      # Tempo característico máximo permitido
THETA_MAX = 50.0    # Deslocamento de fase máximo 

# =============================================================================
# CONSTANTE DE CALIBRAÇÃO DO CHOPPER
# =============================================================================
# Define que o "Tempo Zero" (t=0) acontece no momento em que a lâmina do chopper 
# atinge 35% da amplitude máxima do sinal. Por ser uma porcentagem, isso se 
# autoajusta para amostras claras ou escuras, achando sempre a mesma posição geométrica!
CALIBRACAO_LAMINAS_CHOPPER = 0.25 

# =============================================================================
# 2. FUNÇÕES DA FÍSICA DE LENTE TÉRMICA
# =============================================================================
def calcular_parametros_descasamento():
    """Calcula as constantes do modelo de Shen dependentes apenas da geometria da bancada."""
    V = Z / Z_OP
    m = M_RATIO
    num = 2 * m * V
    den_a = ((1 + 2 * m)**2 + V**2) / 2
    den_b = 1 + 2 * m + V**2
    return num, den_a, den_b

def modelo_shen_v3(t, theta, tc, num, den_a, den_b):
    """
    Equação de Shen que traça a curva vermelha do ajuste.
    theta: Força da lente térmica (quanto esquentou).
    tc: Tempo característico (quão rápido a lente se forma).
    """
    # Ignora erro de divisão por zero matematicamente para o t=0
    with np.errstate(divide='ignore'):
        term_tc_over_time = np.where(t > 0, tc / t, np.inf)
    
    denominador = den_a * term_tc_over_time + den_b
    atan_arg = num / denominador
    return (1 - (theta / 2) * np.arctan(atan_arg))**2

def calcular_eficiencia_e_temperatura(theta, tc, tempo_array):
    """Usa o theta encontrado para calcular o calor e a Eficiência Quântica de Fluorescência."""
    P_in_real = P_IN * (1.0 - PERDA_CUBETA) # Potência real que entra no líquido
    P_abs = P_in_real * (1 - 10**(-A_ABS))  # Potência total absorvida pelas moléculas
    
    # phi = Fração não-radiativa (o percentual de energia que virou calor)
    phi = (np.abs(theta) * LAMBDA_EX * KAPPA) / (P_abs * np.abs(DNDT))
    
    # eta = O restante da energia (que não virou calor) é a Fluorescência (corrigida pelos lambdas)
    eta = (1.0 - phi) * (LAMBDA_EM / LAMBDA_EX)
    
    # delta_T = Evolução do aumento da temperatura no centro do laser
    delta_T_array = (P_abs * phi) / (4 * np.pi * L * KAPPA) * np.log(1 + (2 * tempo_array) / tc)
    
    return eta * 100, delta_T_array  

# =============================================================================
# 3. TRATAMENTO DO SINAL DO DAQ (OSCILOSCÓPIO)
# =============================================================================
def ler_dados_robusto(caminho):
    """Lê o ficheiro CSV ignorando textos, cabeçalhos do osciloscópio e erros de vírgula."""
    try:
        df = pd.read_csv(caminho, header=None, sep=None, engine='python')
        # Substitui vírgulas por pontos e força conversão para número
        df_num = df.astype(str).apply(lambda x: x.str.replace(',', '.')).apply(pd.to_numeric, errors='coerce').dropna(how='any')
        
        # Se restaram poucos dados, tenta ler pulando o cabeçalho longo do osciloscópio
        if len(df_num) < 10:
            df = pd.read_csv(caminho, skiprows=15, header=None, sep=None, engine='python')
            df_num = df.astype(str).apply(lambda x: x.str.replace(',', '.')).apply(pd.to_numeric, errors='coerce').dropna(how='any')
            
        cols = [col for col in df_num.columns if df_num[col].notna().sum() > 0]
        return df_num[cols[0]].to_numpy(), df_num[cols[1]].to_numpy()
    except Exception as e:
        print(f"[ERRO] Falha ao ler dados numéricos: {e}")
        return None, None

def isolar_pulso_robusto(tempo, sinal):
    """
    Encontra o pulso da Lente Térmica cravando o t=0 usando o método dos 35%.
    Ignora ruídos ao encontrar a "subida real" usando a derivada (np.diff).
    """
    v_min, v_max = np.min(sinal), np.max(sinal)
    amplitude = v_max - v_min
    
    if amplitude < 1e-4: return tempo, sinal 
        
    limiar_50 = v_min + 0.5 * amplitude
    estado = (sinal > limiar_50).astype(int)
    transicoes = np.diff(estado) # Encontra os momentos em que o sinal cruza 50%
    
    bordas_subida = np.where(transicoes == 1)[0]
    bordas_descida = np.where(transicoes == -1)[0]
    
    if len(bordas_subida) == 0 or len(bordas_descida) == 0: return tempo, sinal
        
    idx_inicio_50 = bordas_subida[0]
    idx_fim = bordas_descida[bordas_descida > idx_inicio_50]
    idx_fim = idx_fim[0] if len(idx_fim) > 0 else len(sinal) - 1

    # Desce do meio da subida (50%) para trás até bater exatamente nos 35% configurados
    limiar_I0 = v_min + (CALIBRACAO_LAMINAS_CHOPPER * amplitude)
    
    idx_corte = idx_inicio_50
    while idx_corte > 0 and sinal[idx_corte] > limiar_I0:
        idx_corte -= 1
        
    # Recorta o sinal apenas da área útil da Lente Térmica
    t_corte = tempo[idx_corte:idx_fim]
    s_corte = sinal[idx_corte:idx_fim]
    t_corte = t_corte - t_corte[0] # Zera o relógio do tempo
    
    return t_corte, s_corte

# =============================================================================
# 4. LOOP PRINCIPAL (EXECUÇÃO E GERAÇÃO DE GRÁFICOS)
# =============================================================================
def main():
    # Inicializa o Tkinter de forma rápida para não conflitar com o Spyder
    root = tk.Tk()
    root.withdraw() # Oculta a janela principal vazia
    
    print("Selecione os ficheiros de dados (.csv)...")
    arquivos = filedialog.askopenfilenames(
        title="Selecione os ficheiros CSV",
        filetypes=(("Ficheiros CSV", "*.csv"), ("Todos os ficheiros", "*.*"))
    )
    
    # Destrói o Tkinter COMPLETAMENTE assim que o utilizador escolhe os ficheiros.
    # É isso que evita o congelamento do Spyder depois!
    root.destroy() 
    
    if not arquivos:
        print("Nenhum ficheiro selecionado. A encerrar.")
        return

    # Chama as funções geométricas antes do loop
    num, den_a, den_b = calcular_parametros_descasamento()
    fit_func = lambda t, th, tc: modelo_shen_v3(t, th, tc, num, den_a, den_b)
    chute_inicial = [0.5, 5e-3] # Valores iniciais chutados para o curve_fit iniciar a busca
    limites = ([-THETA_MAX, TC_MIN], [THETA_MAX, TC_MAX])

    # Percorre cada arquivo CSV selecionado
    for caminho_arquivo in arquivos:
        nome_arquivo = caminho_arquivo.split('/')[-1]
        print(f"\nA processar: {nome_arquivo}")
        
        tempo_bruto, sinal_bruto = ler_dados_robusto(caminho_arquivo)
        if tempo_bruto is None or len(tempo_bruto) == 0: continue

        # Isola e normaliza o pulso para a matemática de Shen
        t_util, s_util = isolar_pulso_robusto(tempo_bruto, sinal_bruto)
        I0 = s_util[0]
        if I0 <= 0: I0 = 1e-5 
        s_norm = s_util / I0
        
        # Converte de milissegundos para segundos se necessário
        t_sec = t_util / 1000.0 if max(t_util) > 10 else t_util

        try:
            # O Algoritmo matemático encontra a curva que melhor "abraça" os pontos experimentais
            popt, pcov = curve_fit(fit_func, t_sec, s_norm, p0=chute_inicial, bounds=limites)
            theta_fit, tc_fit = popt
            s_fit = fit_func(t_sec, theta_fit, tc_fit)
            
            # Com o Theta encontrado, extrai os resultados físicos reais
            eta_pct, delta_T_array = calcular_eficiencia_e_temperatura(theta_fit, tc_fit, t_sec)
            delta_T_max = np.max(delta_T_array)
            
        except Exception as e:
            print(f"Falha no ajuste matemático para {nome_arquivo}: {e}")
            continue

        # =====================================================================
        # 5. PLOTAGEM DOS RESULTADOS DA LENTE TÉRMICA
        # =====================================================================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.canvas.manager.set_window_title(f"Ajuste - {nome_arquivo}")
        
        t_plot = t_sec * 1000 # Converte de volta para ms para visualização bonita
        
        # Gráfico Esquerdo: Óptica e Ajuste de Shen
        ax1.plot(t_plot, s_norm, '.', color='gray', alpha=0.7, label='Dados Experimentais')
        ax1.plot(t_plot, s_fit, '-', color='red', linewidth=2.5, label='Fit Físico (Shen)')
        ax1.set_title(f"Ajuste Óptico\n$\\theta$={np.abs(theta_fit):.4f} | $t_c$={tc_fit*1000:.2f} ms\nTensão Inicial $I(0)$: {I0:.4f} V")
        ax1.set_xlabel("Tempo (ms)")
        ax1.set_ylabel("I(t)/I(0)")
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()
        
        # Gráfico Direito: Temperatura Dinâmica e Rendimento
        ax2.plot(t_plot, delta_T_array, '-', color='blue', linewidth=2)
        ax2.set_title(f"Temperatura & Eficiência\n$\\eta$={eta_pct:.1f}% | $\\Delta T_{{max}}$={delta_T_max:.3f}°C")
        ax2.set_xlabel("Tempo (ms)")
        ax2.set_ylabel("$\\Delta T$ (°C)")
        ax2.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # =====================================================================
        # NOVA SOLUÇÃO ANTI-TRAVAMENTO PARA SPYDER
        # =====================================================================
        plt.show(block=False) # Mostra o gráfico, mas não congela o Spyder
        
        # Este laço pequeno força o Python a atualizar a tela enquanto o gráfico
        # existir. Quando você clica em fechar, a figura deixa de existir, o laço
        # quebra e o código continua para a próxima amostra sem travar nada!
        while plt.fignum_exists(fig.number):
            plt.pause(0.1) 
            
        # O print só sai no console DEPOIS que você fechar a janela
        print(f"-> Eficiência (η): {eta_pct:.1f}% | I(0): {I0:.4f}V | Theta: {np.abs(theta_fit):.4f}")

    print("\nProcessamento concluído!")

# Executa o programa
if __name__ == "__main__":
    main()