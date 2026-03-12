import board
import busio
import numpy as np
import collections
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# --- 1. CONFIGURAÇÃO DO HARDWARE (ADS1115) ---
try:
    i2c = busio.I2C(board.SCL, board.SDA)
    ads = ADS.ADS1115(i2c)
    ads.gain = 1  # Faixa de +/- 4.096V (ideal para ler 3.3V)
    chan = AnalogIn(ads, ADS.P0)
except Exception as e:
    print(f"Erro ao configurar ADS1115: {e}")
    exit()

# --- 2. CONFIGURAÇÃO DO GRÁFICO (OSCILOSCÓPIO) ---
fig, ax = plt.subplots()
plt.title("Osciloscópio Digital - Raspberry Pi + ADS1115")
plt.xlabel("Tempo de Amostragem (Relativo)")
plt.ylabel("Tensão (V)")

# Parâmetros da tela
max_pontos = 100            # Quantidade de amostras visíveis na tela
intervalo_leitura = 20      # Milissegundos entre leituras (ajusta a velocidade)

# Eixo X fixo (representando a grade da tela)
x_vals = np.linspace(0, 10, max_pontos)

# Buffer circular para os dados do eixo Y
y_data = collections.deque([0.0] * max_pontos, maxlen=max_pontos)

# Configuração visual do eixo
ax.set_xlim(0, 10)
ax.set_ylim(-0.1, 4.0)      # Focado na faixa de 0V a 3.3V
ax.grid(True, linestyle='--', color='gray', alpha=0.5)

# Linha do gráfico (Estilo clássico de osciloscópio)
linha, = ax.plot(x_vals, list(y_data), lw=2, color='lime')

# Remover numeração do eixo X para parecer osciloscópio puro (opcional)
ax.set_xticks([])

# --- 3. LÓGICA DE ATUALIZAÇÃO ---
def atualizar(frame):
    try:
        # Leitura real do sensor
        tensao_atual = chan.voltage
        
        # Adiciona ao buffer (automaticamente descarta o mais antigo)
        y_data.append(tensao_atual)
        
        # Atualiza o gráfico com os dados atuais do buffer
        linha.set_ydata(list(y_data))
        
    except Exception as e:
        print(f"Falha na leitura: {e}")
        
    return linha,

# --- 4. INICIALIZAÇÃO DA ANIMAÇÃO ---
# interval=20ms significa que tentaremos ler ~50 vezes por segundo
ani = animation.FuncAnimation(
    fig, 
    atualizar, 
    interval=intervalo_leitura, 
    blit=True, 
    cache_frame_data=False
)

print("Iniciando osciloscópio... Feche a janela para parar.")
plt.show()