import time
import threading
import queue
import numpy as np
import pyqtgraph as pg

# O pyqtgraph descobre sozinho se você está usando PyQt5 ou PyQt6 e importa o correto
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

# --- Bibliotecas do Hardware ---
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import adafruit_ads1x15.ads1x15 as ads1x15
from collections import deque

# --- Configuração do I2C e ADS1115 ---
try:
    i2c = busio.I2C(board.SCL, board.SDA)
    ads = ADS.ADS1115(i2c, data_rate=860) 
    chan = AnalogIn(ads, ads1x15.Pin.A0) 
except Exception as e:
    print(f"Erro ao inicializar: {e}")
    exit()

# --- Configurações Iniciais ---
frequencia_amostragem = 860
TEMPO_MAXIMO = 2.0
tamanho_buffer = int(TEMPO_MAXIMO * frequencia_amostragem) * 3
buffer_interno = deque([0.0] * tamanho_buffer, maxlen=tamanho_buffer)

fila_dados = queue.Queue()
rodando = True

# --- Thread de Leitura ---
def ler_adc_continuamente():
    periodo_alvo = 1.0 / frequencia_amostragem 
    while rodando:
        inicio = time.perf_counter()
        try:
            fila_dados.put(chan.voltage)
        except Exception:
            pass
        tempo_gasto = time.perf_counter() - inicio
        if tempo_gasto < periodo_alvo:
            time.sleep(periodo_alvo - tempo_gasto)

thread_adc = threading.Thread(target=ler_adc_continuamente, daemon=True)
thread_adc.start()

# --- Classe Principal da Interface Gráfica ---
class OsciloscopioApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Osciloscópio PyQtGraph")
        self.resize(800, 480) # Resolução baseada em telas de 7"
        self.setStyleSheet("background-color: #1e1e1e; color: white;")

        # Variáveis de Controle
        self.tempo_visivel = 0.5
        self.tensao_escala = 4.0
        self.trigger_ativado = False
        self.nivel_trigger = 2.0
        
        self.max_pontos = int(self.tempo_visivel * frequencia_amostragem)
        self.y_data_tela = np.zeros(self.max_pontos)

        self.setup_ui()
        
        # Timer de Atualização (roda a 30 FPS para poupar CPU)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.atualizar_grafico)
        self.timer.start(33) 

    def setup_ui(self):
        layout_principal = QtWidgets.QVBoxLayout(self)

        # 1. Área do Gráfico
        self.plot_widget = pg.PlotWidget(title="Leitura do Fotodiodo (ADS1115 via I2C)")
        self.plot_widget.setYRange(-0.2, self.tensao_escala)
        self.plot_widget.setXRange(0, self.tempo_visivel)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.5)
        self.plot_widget.setLabel('left', 'Tensão (V)')
        self.plot_widget.setLabel('bottom', 'Tempo Estimado (s)')
        
        self.curva_sinal = self.plot_widget.plot(pen=pg.mkPen('#00FF00', width=2)) # Cor Lime
        
        # Linha do Trigger (invisível por padrão)
        self.linha_trigger_h = pg.InfiniteLine(angle=0, pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DashLine))
        self.plot_widget.addItem(self.linha_trigger_h)
        self.linha_trigger_h.setVisible(False)

        layout_principal.addWidget(self.plot_widget, stretch=3)

        # 2. Painel de Informações (RMS e Frequência)
        self.label_info = QtWidgets.QLabel("Tensão RMS: 0.000 V | Frequência: 0.0 Hz")
        self.label_info.setStyleSheet("font-size: 16px; font-weight: bold; color: lime; background-color: black; padding: 5px; border: 1px solid lime; border-radius: 5px;")
        self.label_info.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter if hasattr(QtCore.Qt, 'AlignmentFlag') else QtCore.Qt.AlignCenter)
        layout_principal.addWidget(self.label_info)

        # 3. Painel de Controles (Sliders e Botão)
        painel_controles = QtWidgets.QGridLayout()
        
        estilo_slider = """
            QSlider::groove:horizontal { border: 1px solid #999999; height: 15px; background: #333333; margin: 2px 0; border-radius: 4px; }
            QSlider::handle:horizontal { background: cyan; border: 1px solid #5c5c5c; width: 25px; margin: -5px 0; border-radius: 4px; }
        """

        # Slider Tempo
        painel_controles.addWidget(QtWidgets.QLabel("Tempo (s):"), 0, 0)
        self.slider_tempo = self.criar_slider(1, 20, int(self.tempo_visivel * 10), estilo_slider)
        self.slider_tempo.valueChanged.connect(self.ao_mudar_tempo)
        painel_controles.addWidget(self.slider_tempo, 0, 1)
        self.lbl_val_tempo = QtWidgets.QLabel(f"{self.tempo_visivel} s")
        painel_controles.addWidget(self.lbl_val_tempo, 0, 2)

        # Slider Tensão Máxima
        painel_controles.addWidget(QtWidgets.QLabel("Tensão Máx (V):"), 1, 0)
        self.slider_tensao = self.criar_slider(5, 50, int(self.tensao_escala * 10), estilo_slider.replace('cyan', 'magenta'))
        self.slider_tensao.valueChanged.connect(self.ao_mudar_tensao)
        painel_controles.addWidget(self.slider_tensao, 1, 1)
        self.lbl_val_tensao = QtWidgets.QLabel(f"{self.tensao_escala} V")
        painel_controles.addWidget(self.lbl_val_tensao, 1, 2)

        # Slider Trigger
        painel_controles.addWidget(QtWidgets.QLabel("Nível Trigger (V):"), 2, 0)
        self.slider_trigger = self.criar_slider(1, 40, int(self.nivel_trigger * 10), estilo_slider.replace('cyan', 'yellow'))
        self.slider_trigger.valueChanged.connect(self.ao_mudar_nivel_trigger)
        painel_controles.addWidget(self.slider_trigger, 2, 1)
        self.lbl_val_trigger = QtWidgets.QLabel(f"{self.nivel_trigger} V")
        painel_controles.addWidget(self.lbl_val_trigger, 2, 2)

        # Botão Trigger
        self.btn_trigger = QtWidgets.QPushButton("Trigger: OFF")
        self.btn_trigger.setStyleSheet("background-color: darkred; color: white; font-weight: bold; font-size: 14px; padding: 10px; border-radius: 5px;")
        self.btn_trigger.clicked.connect(self.alternar_trigger)
        painel_controles.addWidget(self.btn_trigger, 0, 3, 3, 1) # Ocupa as 3 linhas

        layout_principal.addLayout(painel_controles, stretch=1)

    def criar_slider(self, minimo, maximo, atual, estilo):
        slider = QtWidgets.QSlider()
        slider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        slider.setMinimum(minimo)
        slider.setMaximum(maximo)
        slider.setValue(atual)
        slider.setStyleSheet(estilo)
        return slider

    # --- Funções de Interface ---
    def ao_mudar_tempo(self, val):
        self.tempo_visivel = val / 10.0
        self.lbl_val_tempo.setText(f"{self.tempo_visivel:.1f} s")
        self.max_pontos = int(self.tempo_visivel * frequencia_amostragem)
        self.y_data_tela = np.zeros(self.max_pontos)
        self.plot_widget.setXRange(0, self.tempo_visivel)

    def ao_mudar_tensao(self, val):
        self.tensao_escala = val / 10.0
        self.lbl_val_tensao.setText(f"{self.tensao_escala:.1f} V")
        self.plot_widget.setYRange(-0.2, self.tensao_escala)

    def ao_mudar_nivel_trigger(self, val):
        self.nivel_trigger = val / 10.0
        self.lbl_val_trigger.setText(f"{self.nivel_trigger:.1f} V")
        self.linha_trigger_h.setPos(self.nivel_trigger)

    def alternar_trigger(self):
        self.trigger_ativado = not self.trigger_ativado
        if self.trigger_ativado:
            self.btn_trigger.setText("Trigger: ON")
            self.btn_trigger.setStyleSheet("background-color: darkgreen; color: white; font-weight: bold; font-size: 14px; padding: 10px; border-radius: 5px;")
            self.linha_trigger_h.setVisible(True)
            self.linha_trigger_h.setPos(self.nivel_trigger)
        else:
            self.btn_trigger.setText("Trigger: OFF")
            self.btn_trigger.setStyleSheet("background-color: darkred; color: white; font-weight: bold; font-size: 14px; padding: 10px; border-radius: 5px;")
            self.linha_trigger_h.setVisible(False)

    # --- Loop de Atualização ---
    def atualizar_grafico(self):
        novos_dados = False
        
        q_size = fila_dados.qsize()
        for _ in range(q_size):
            try:
                buffer_interno.append(fila_dados.get_nowait())
                novos_dados = True
            except queue.Empty:
                break
                
        if novos_dados and len(buffer_interno) > self.max_pontos:
            arr = np.array(buffer_interno)
            atualizar_tela = False
            
            PONTOS_PRE = int(self.max_pontos * 0.1)
            PONTOS_POS = self.max_pontos - PONTOS_PRE

            if self.trigger_ativado:
                condicao_subida = (arr[:-1] < self.nivel_trigger) & (arr[1:] >= self.nivel_trigger)
                indices_trigger = np.where(condicao_subida)[0]
                triggers_validos = [i for i in indices_trigger if (i >= PONTOS_PRE) and ((len(arr) - i - 1) >= PONTOS_POS)]
                
                if triggers_validos:
                    idx_trigger = triggers_validos[-1] + 1 
                    novo_y = arr[idx_trigger - PONTOS_PRE : idx_trigger + PONTOS_POS]
                    atualizar_tela = True
            else:
                novo_y = arr[-self.max_pontos:]
                atualizar_tela = True

            if atualizar_tela and len(novo_y) == len(self.y_data_tela):
                self.y_data_tela[:] = novo_y
                
                # Eixo X simulado pelo tempo
                x_data = np.linspace(0, self.tempo_visivel, self.max_pontos)
                self.curva_sinal.setData(x=x_data, y=self.y_data_tela)

                # Cálculos (RMS e Freq)
                tensao_rms = np.sqrt(np.mean(np.square(self.y_data_tela)))
                frequencia_hz = 0.0
                
                acima_limiar = self.y_data_tela > 2.0 
                mudancas = np.diff(acima_limiar.astype(int))
                bordas_subida = np.where(mudancas == 1)[0]
                if len(bordas_subida) > 1:
                    periodos_pontos = np.diff(bordas_subida)
                    frequencia_hz = frequencia_amostragem / np.mean(periodos_pontos)

                self.label_info.setText(f"Tensão RMS: {tensao_rms:.3f} V | Frequência Estimada: {frequencia_hz:.1f} Hz")

    def closeEvent(self, event):
        global rodando
        rodando = False
        event.accept()

# --- Execução ---
if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    janela = OsciloscopioApp()
    janela.show()
    # O comando app.exec() já está no formato atualizado para evitar o erro anterior
    app.exec()
