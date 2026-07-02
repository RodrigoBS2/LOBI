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

Z_OP = 1.9e-3       
Z = 1.05e-2         # CORREÇÃO: Z Positivo (PÓS-FOCAL) para permitir que a curva caia.
L = 1e-3            
M_RATIO = 1.0       
PERDA_CUBETA = 0.10 

DNDT = -3.94e-4     
KAPPA = 0.171       

P_IN = 17e-3        
A_ABS = 0.3567      
LAMBDA_EX = 532e-9  
LAMBDA_EM = 560e-9  

TC_MIN = 1e-4       
TC_MAX = 50e-3      
THETA_MAX = 50.0    

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
    P_abs = P_in_real * (1 - 10**(-A_ABS))  
    
    phi = (np.abs(theta) * LAMBDA_EX * KAPPA) / (P_abs * np.abs(DNDT))
    eta = (1.0 - phi) * (LAMBDA_EM / LAMBDA_EX)
    
    delta_T_array = (P_abs * phi) / (4 * np.pi * L * KAPPA) * np.log(1 + (2 * tempo_array) / tc)
    
    return eta * 100, delta_T_array  

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
        print(f"[ERRO] Falha ao ler dados numéricos: {e}")
        return None, None

def isolar_pulso_robusto(tempo, sinal):
    """
    Restauração do algoritmo de derivada. Pula a subida mecânica do chopper 
    e define o PICO do sinal como I(0), isolando apenas o resfriamento real.
    """
    v_min, v_max = np.min(sinal), np.max(sinal)
    amplitude = v_max - v_min
    if amplitude < 1e-4: return tempo, sinal 

    limiar_50 = v_min + 0.5 * amplitude
    transicoes = np.diff((sinal > limiar_50).astype(int))

    bordas_subida = np.where(transicoes == 1)[0]
    bordas_descida = np.where(transicoes == -1)[0]

    if len(bordas_subida) == 0: return tempo, sinal
    idx_subida = bordas_subida[0]
    
    descidas_validas = bordas_descida[bordas_descida > idx_subida]
    idx_fim = descidas_validas[0] if len(descidas_validas) > 0 else len(sinal) - 1

    # Usa a derivada para encontrar quando a hélice finalmente parou de subir
    sinal_suave = np.convolve(sinal, np.ones(5)/5.0, mode='same')
    derivada = np.gradient(sinal_suave)

    inicio_busca = max(0, idx_subida - 10)
    fim_busca = min(len(derivada), idx_subida + 20)
    pico_derivada = np.max(derivada[inicio_busca:fim_busca])

    idx_inicio_real = idx_subida
    while idx_inicio_real < idx_fim and derivada[idx_inicio_real] > 0.15 * pico_derivada:
        idx_inicio_real += 1

    idx_inicio_real += 2 # Foge da turbulência da borda

    t_corte = tempo[idx_inicio_real:idx_fim]
    s_corte = sinal[idx_inicio_real:idx_fim]
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
        return

    num, den_a, den_b = calcular_parametros_descasamento()
    fit_func = lambda t, th, tc: modelo_shen_v3(t, th, tc, num, den_a, den_b)
    limites = ([-THETA_MAX, TC_MIN], [THETA_MAX, TC_MAX])

    for caminho_arquivo in arquivos:
        nome_arquivo = caminho_arquivo.split('/')[-1]
        print(f"\nA processar: {nome_arquivo}")
        
        tempo_bruto, sinal_bruto = ler_dados_robusto(caminho_arquivo)
        if tempo_bruto is None or len(tempo_bruto) == 0: continue

        t_util, s_util = isolar_pulso_robusto(tempo_bruto, sinal_bruto)
        I0 = s_util[0]
        if I0 <= 0: I0 = 1e-5 
        s_norm = s_util / I0
        
        t_sec = t_util / 1000.0 if max(t_util) > 10 else t_util

        # CHUTE INICIAL DINÂMICO PARA DECAIMENTO (PÓS-FOCAL)
        s_inf = np.mean(s_norm[-10:]) # Intensidade no fundo da curva
        val_atan = np.abs(np.arctan(num / den_b))
        if val_atan == 0: val_atan = 1e-5
        
        theta_guess = 2 * (1 - np.sqrt(np.abs(s_inf))) / val_atan
        if theta_guess < 0.1: theta_guess = 0.5
        if theta_guess > THETA_MAX: theta_guess = THETA_MAX / 2.0
        
        chute_inicial = [theta_guess, 5e-3]

        try:
            popt, pcov = curve_fit(fit_func, t_sec, s_norm, p0=chute_inicial, bounds=limites)
            theta_fit, tc_fit = popt
            s_fit = fit_func(t_sec, theta_fit, tc_fit)
            
            eta_pct, delta_T_array = calcular_eficiencia_e_temperatura(theta_fit, tc_fit, t_sec)
            delta_T_max = np.max(delta_T_array)
            
        except Exception as e:
            print(f"Falha no ajuste matemático: {e}")
            continue

        # =====================================================================
        # 5. PLOTAGEM
        # =====================================================================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.canvas.manager.set_window_title(f"Ajuste - {nome_arquivo}")
        t_plot = t_sec * 1000 
        
        ax1.plot(t_plot, s_norm, '.', color='gray', alpha=0.7, label='Dados Experimentais')
        ax1.plot(t_plot, s_fit, '-', color='red', linewidth=2.5, label='Fit Físico (Shen)')
        ax1.set_title(f"Ajuste Óptico\n$\\theta$={np.abs(theta_fit):.4f} | $t_c$={tc_fit*1000:.2f} ms\nTensão PICO $I(0)$: {I0:.4f} V")
        ax1.set_xlabel("Tempo (ms)")
        ax1.set_ylabel("I(t)/I(0)")
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()
        
        ax2.plot(t_plot, delta_T_array, '-', color='blue', linewidth=2)
        ax2.set_title(f"Temperatura & Eficiência\n$\\eta$={eta_pct:.1f}% | $\\Delta T_{{max}}$={delta_T_max:.3f}°C")
        ax2.set_xlabel("Tempo (ms)")
        ax2.set_ylabel("$\\Delta T$ (°C)")
        ax2.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        plt.show(block=False) 
        
        while plt.fignum_exists(fig.number):
            plt.pause(0.1) 
            
        print(f"-> Eficiência (η): {eta_pct:.1f}% | I(0): {I0:.4f}V | Theta: {np.abs(theta_fit):.4f}")

    print("\nProcessamento concluído!")

if __name__ == "__main__":
    main()