# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:06:11 2026

@author: Flávia Eduarda
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog

'''

# Adicione nos parâmetros iniciais:
THETA_SOLVENTE = 0.015 # Substitua pelo valor ajustado da sua cubeta só com Etanol

def calcular_eficiencia_e_temperatura(theta, tc, tempo_array):
    P_in_real = P_IN * (1.0 - PERDA_CUBETA)
    alfa_m = ALFA * 100 
    P_abs = P_in_real * (1 - np.exp(-alfa_m * L))
    
    # SUBTRAÇÃO DO BRANCO (Apenas a força da Lente Térmica da molécula):
    theta_real = np.abs(theta) - THETA_SOLVENTE
    
    # Calcula o calor apenas com o theta real da Rodamina:
    phi = (theta_real * LAMBDA_EX * KAPPA) / (P_abs * np.abs(DNDT))
    eta = (1.0 - phi) * (LAMBDA_EM / LAMBDA_EX)
    
    delta_T_global = (P_abs * phi) / (4 * np.pi * L * KAPPA * (1 + M_RATIO)) * np.log(1 + (2 * tempo_array) / tc)
    
    return eta * 100, delta_T_global, P_abs

'''


# =============================================================================
# 1. PARÂMETROS FÍSICOS CONSTANTES 
# =============================================================================

# -- Parâmetros Geométricos (Montagem Óptica) --
Z_OP = 1.9e-3       # Z_OP (m): Posição confocal da lente.
Z = 1.05e-2         # Z (m): Distância da amostra até a cintura do laser.
L = 1e-3            # L (m): Espessura da amostra (caminho óptico).
M_RATIO = 1.0       # m: Grau de descasamento (Feixe Único = 1.0).
PERDA_CUBETA = 0.10 # Perda de Fresnel (10%).

# -- Propriedades do Solvente --
DNDT = -3.94e-4     # dn/dT (K^-1)
KAPPA = 0.171       # kappa (W/m.K)

# -- Propriedades da Amostra e Laser --
P_IN = 17.5e-3        # P_in (W): Potência bruta do laser.
ALFA = 34.94         # alpha (cm^-1): Coeficiente de absorção linear (Mude este valor conforme a amostra!)

LAMBDA_EX = 532e-9  # Lambda_ex (m)
LAMBDA_EM = 560e-9  # Lambda_em (m)

# -- Limites Matemáticos do Ajuste --
TC_MIN = 1e-4       
TC_MAX = 50e-3      
THETA_MAX = 50.0    

# Constante de Calibração
CALIBRACAO_LAMINAS_CHOPPER = 0.25 

# =============================================================================
# 2. FUNÇÕES DA FÍSICA DE LENTE TÉRMICA
# =============================================================================
def calcular_parametros_descasamento():
    V = Z / Z_OP
    m = M_RATIO
    num = 2 * m * V
    den_a = ((1 + 2 * m)**2 + V**2) / 2
    den_b = 1 + 2 * m + V**2
    return num, den_a, den_b

def modelo_shen_v3(t, theta, tc, num, den_a, den_b):
    with np.errstate(divide='ignore'):
        term_tc_over_time = np.where(t > 0, tc / t, np.inf)
    
    denominador = den_a * term_tc_over_time + den_b
    atan_arg = num / denominador
    return (1 - (theta / 2) * np.arctan(atan_arg))**2

def calcular_eficiencia_e_temperatura(theta, tc, tempo_array):
    P_in_real = P_IN * (1.0 - PERDA_CUBETA)
    
    # O Python multiplica o ALFA por 100 para converter de cm^-1 para m^-1 internamente!
    # Assim as unidades batem perfeitamente com o L em metros (SI).
    alfa_m = ALFA * 100 
    P_abs = P_in_real * (1 - np.exp(-alfa_m * L))
    
    phi = (np.abs(theta) * LAMBDA_EX * KAPPA) / (P_abs * np.abs(DNDT))
    eta = (1.0 - phi) * (LAMBDA_EM / LAMBDA_EX)
    
    # Correção da Temperatura Global (Feixe Único)
    # A equação usa o fator (1 + m) no denominador. Sendo m=1, o 4*pi clássico vira 8*pi.
    # Isso reflete a média espacial da lente sentida pelo próprio feixe!
    delta_T_global = (P_abs * phi) / (4 * np.pi * L * KAPPA * (1 + M_RATIO)) * np.log(1 + (2 * tempo_array) / tc)
    
    return eta * 100, delta_T_global, P_abs  

# =============================================================================
# 3. TRATAMENTO DO SINAL DO DAQ
# =============================================================================
def ler_dados_robusto(caminho):
    try:
        df = pd.read_csv(caminho, header=None, sep=None, engine='python')
        df_num = df.astype(str).apply(lambda x: x.str.replace(',', '.')).apply(pd.to_numeric, errors='coerce').dropna(how='any')
        if len(df_num) < 10:
            df = pd.read_csv(caminho, skiprows=15, header=None, sep=None, engine='python')
            df_num = df.astype(str).apply(lambda x: x.str.replace(',', '.')).apply(pd.to_numeric, errors='coerce').dropna(how='any')
            
        cols = [col for col in df_num.columns if df_num[col].notna().sum() > 0]
        return df_num[cols[0]].to_numpy(), df_num[cols[1]].to_numpy()
    except Exception as e:
        print(f"[ERRO] Falha ao ler dados: {e}")
        return None, None

def isolar_pulso_robusto(tempo, sinal):
    v_min, v_max = np.min(sinal), np.max(sinal)
    amplitude = v_max - v_min
    if amplitude < 1e-4: return tempo, sinal 
        
    limiar_50 = v_min + 0.5 * amplitude
    estado = (sinal > limiar_50).astype(int)
    transicoes = np.diff(estado)
    
    bordas_subida = np.where(transicoes == 1)[0]
    bordas_descida = np.where(transicoes == -1)[0]
    
    if len(bordas_subida) == 0 or len(bordas_descida) == 0: return tempo, sinal
        
    idx_inicio_50 = bordas_subida[0]
    idx_fim = bordas_descida[bordas_descida > idx_inicio_50]
    idx_fim = idx_fim[0] if len(idx_fim) > 0 else len(sinal) - 1

    limiar_I0 = v_min + (CALIBRACAO_LAMINAS_CHOPPER * amplitude)
    
    idx_corte = idx_inicio_50
    while idx_corte > 0 and sinal[idx_corte] > limiar_I0:
        idx_corte -= 1
        
    t_corte = tempo[idx_corte:idx_fim]
    s_corte = sinal[idx_corte:idx_fim]
    t_corte = t_corte - t_corte[0] 
    
    return t_corte, s_corte

# =============================================================================
# 4. LOOP PRINCIPAL
# =============================================================================
def main():
    root = tk.Tk()
    root.withdraw() 
    
    print("Selecione os ficheiros de dados (.csv)...")
    arquivos = filedialog.askopenfilenames(
        title="Selecione os ficheiros CSV",
        filetypes=(("Ficheiros CSV", "*.csv"), ("Todos os ficheiros", "*.*"))
    )
    root.destroy() 
    
    if not arquivos:
        print("Nenhum ficheiro selecionado.")
        return

    num, den_a, den_b = calcular_parametros_descasamento()
    fit_func = lambda t, th, tc: modelo_shen_v3(t, th, tc, num, den_a, den_b)
    chute_inicial = [0.5, 5e-3] 
    limites = ([-THETA_MAX, TC_MIN], [THETA_MAX, TC_MAX])

    for caminho_arquivo in arquivos:
        nome_arquivo = caminho_arquivo.split('/')[-1]
        print(f"\n{'='*50}")
        print(f"A processar amostra: {nome_arquivo}")
                
        tempo_bruto, sinal_bruto = ler_dados_robusto(caminho_arquivo)
        if tempo_bruto is None or len(tempo_bruto) == 0: continue

        t_util, s_util = isolar_pulso_robusto(tempo_bruto, sinal_bruto)
        I0 = s_util[0]
        if I0 <= 0: I0 = 1e-5 
        s_norm = s_util / I0
        
        t_sec = t_util / 1000.0 if max(t_util) > 10 else t_util

        try:
            popt, pcov = curve_fit(fit_func, t_sec, s_norm, p0=chute_inicial, bounds=limites)
            theta_fit, tc_fit = popt
            s_fit = fit_func(t_sec, theta_fit, tc_fit)
            
            # Cálculo usando o ALFA constante definido no topo do programa
            eta_pct, delta_T_array, P_abs_calculada = calcular_eficiencia_e_temperatura(theta_fit, tc_fit, t_sec)
            delta_T_max = np.max(delta_T_array)
            
        except Exception as e:
            print(f"Falha no ajuste matemático: {e}")
            continue

        # =====================================================================
        # 5. PLOTAGEM DOS RESULTADOS
        # =====================================================================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.canvas.manager.set_window_title(f"Ajuste - {nome_arquivo}")
        
        t_plot = t_sec * 1000 
        
        ax1.plot(t_plot, s_norm, '.', color='gray', alpha=0.7, label='Dados Experimentais')
        ax1.plot(t_plot, s_fit, '-', color='red', linewidth=2.5, label='Fit Físico (Shen)')
        ax1.set_title(f"Ajuste Óptico\n$\\theta$={np.abs(theta_fit):.4f} | $t_c$={tc_fit*1000:.2f} ms")
        ax1.set_xlabel("Tempo (ms)")
        ax1.set_ylabel("I(t)/I(0)")
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()
        
        ax2.plot(t_plot, delta_T_array, '-', color='blue', linewidth=2)
        ax2.set_title(f"Temperatura Global Média\n$\\eta$={eta_pct:.1f}% | $\\langle\\Delta T\\rangle_{{max}}$={delta_T_max:.3f}°C")
        ax2.set_xlabel("Tempo (ms)")
        ax2.set_ylabel("$\\langle\\Delta T\\rangle$ (°C)")
        ax2.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        plt.show(block=False) 
        
        while plt.fignum_exists(fig.number):
            plt.pause(0.1) 
            
        print(f"Resultados de {nome_arquivo}:")
        print(f"  P_abs : {P_abs_calculada*1000:.2f} mW")
        print(f"  Theta : {np.abs(theta_fit):.4f}")
        print(f"  Eta(η): {eta_pct:.1f}%")

    print("\nProcessamento de todos os ficheiros concluído!")

if __name__ == "__main__":
    main()