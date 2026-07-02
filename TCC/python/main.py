import time
import threading
import serial
import csv
import numpy as np
import pyqtgraph as pg
from datetime import datetime
import os

from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from collections import deque

# --- Configurações da Porta Serial (STM32 - Binário 115200) ---
PORTA_SERIAL = '/dev/ttyUSB0' 
BAUD_RATE = 115200

try:
    ser = serial.Serial(PORTA_SERIAL, BAUD_RATE, timeout=0)
    ser.reset_input_buffer() 
except Exception as e:
    print(f"Erro ao abrir a porta serial {PORTA_SERIAL}: {e}")
    exit()

frequencia_amostragem = 5700             # 5291, 5700, 808
TEMPO_MAXIMO = 1.0 
tamanho_buffer = int(TEMPO_MAXIMO * frequencia_amostragem) * 2

buffer_lock = threading.Lock()
buffer_interno = deque([0.0] * tamanho_buffer, maxlen=tamanho_buffer)

rodando = True

# --- Decodificador de Sincronia Instantânea ---
def ler_serial_continuamente():
    buffer_serial = bytearray()
    while rodando:
        try:
            if ser.in_waiting:
                buffer_serial.extend(ser.read(ser.in_waiting))
                novas_tensoes = []
                
                i = 0
                tamanho = len(buffer_serial)
                
                while i < tamanho - 1:
                    b1 = buffer_serial[i]
                    if b1 >= 128:
                        b2 = buffer_serial[i+1]
                        if b2 < 128:
                            valor_adc = ((b1 & 0x7F) << 7) | b2
                            novas_tensoes.append((valor_adc / 4095.0) * 3.3)
                            i += 2
                        else:
                            i += 1
                    else:
                        i += 1
                        
                del buffer_serial[:i]
                
                if novas_tensoes:
                    with buffer_lock:
                        buffer_interno.extend(novas_tensoes)
            else:
                time.sleep(0.001)
        except Exception:
            pass

thread_serial = threading.Thread(target=ler_serial_continuamente, daemon=True)
thread_serial.start()

# --- Classe Principal da Interface Gráfica ---
class OsciloscopioApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Osciloscópio PyQtGraph - Sistema DAQ")
        self.resize(850, 540) 
        self.setStyleSheet("background-color: #1e1e1e; color: white;")

        self.tempo_visivel = 0.1 
        self.escala_y = 4.000 
        self.offset_y = 0.000 
        
        self.trigger_ativado = False
        self.nivel_trigger = 1.500 
        self.borda_trigger = 'subida' 
        self.media_ativada = False
        self.filtro_ativado = False 
        self.qtd_amostras_media = 16
        self.buffer_ondas = deque(maxlen=self.qtd_amostras_media)
        
        self.max_pontos = int(self.tempo_visivel * frequencia_amostragem)
        self.y_data_tela = np.zeros(self.max_pontos)

        self.setup_ui()
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.atualizar_grafico)
        self.timer.start(50) 

    def setup_ui(self):
        layout_principal = QtWidgets.QVBoxLayout(self)

        self.plot_widget = pg.PlotWidget(title="Sinal do Fotodetector")
        self.plot_widget.setYRange(self.offset_y, self.offset_y + self.escala_y, padding=0)
        self.plot_widget.setXRange(0, self.tempo_visivel, padding=0)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.5)
        self.plot_widget.setLabel('left', 'Tensão (V)')
        self.plot_widget.setLabel('bottom', 'Tempo (s)')
        
        self.curva_sinal = self.plot_widget.plot(pen=pg.mkPen('#00FF00', width=2))
        
        self.linha_trigger_h = pg.InfiniteLine(angle=0, pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DashLine))
        self.plot_widget.addItem(self.linha_trigger_h)
        self.linha_trigger_h.setVisible(False)
        self.linha_trigger_h.setPos(self.nivel_trigger)
        
        PONTOS_PRE = int(self.max_pontos * 0.1)
        tempo_pre = (PONTOS_PRE / frequencia_amostragem)
        self.linha_trigger_v = pg.InfiniteLine(pos=tempo_pre, angle=90, pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DashLine))
        self.plot_widget.addItem(self.linha_trigger_v)
        self.linha_trigger_v.setVisible(False)

        layout_principal.addWidget(self.plot_widget, stretch=3)

        self.label_info = QtWidgets.QLabel("Tensão Média (DC): 0.000 V | Frequência Estimada: 0.0 Hz")
        self.label_info.setStyleSheet("font-size: 16px; font-weight: bold; color: lime; background-color: black; padding: 5px; border: 1px solid lime; border-radius: 5px;")
        self.label_info.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter if hasattr(QtCore.Qt, 'AlignmentFlag') else QtCore.Qt.AlignCenter)
        layout_principal.addWidget(self.label_info)

        painel_controles = QtWidgets.QGridLayout()
        
        estilo_slider = """
            QSlider::groove:horizontal { border: 1px solid #999999; height: 15px; background: #333333; margin: 2px 0; border-radius: 4px; }
            QSlider::handle:horizontal { background: cyan; border: 1px solid #5c5c5c; width: 25px; margin: -5px 0; border-radius: 4px; }
        """

        painel_controles.addWidget(QtWidgets.QLabel("Tempo Visível (s):"), 0, 0)
        self.slider_tempo = self.criar_slider(1, 100, int(self.tempo_visivel * 100), estilo_slider)
        self.slider_tempo.valueChanged.connect(self.ao_mudar_tempo)
        painel_controles.addWidget(self.slider_tempo, 0, 1)
        self.lbl_val_tempo = QtWidgets.QLabel(f"{self.tempo_visivel:.2f} s")
        painel_controles.addWidget(self.lbl_val_tempo, 0, 2)
        
        self.btn_trigger = QtWidgets.QPushButton("Trigger: OFF")
        self.btn_trigger.setStyleSheet("background-color: darkred; color: white; font-weight: bold; font-size: 14px; padding: 5px; border-radius: 5px;")
        self.btn_trigger.clicked.connect(self.alternar_trigger)
        painel_controles.addWidget(self.btn_trigger, 0, 3) 

        painel_controles.addWidget(QtWidgets.QLabel("Zoom Y (Tamanho):"), 1, 0)
        self.slider_escala = self.criar_slider(10, 4000, int(self.escala_y * 1000), estilo_slider.replace('cyan', 'magenta'))
        self.slider_escala.valueChanged.connect(self.ao_mudar_escala)
        painel_controles.addWidget(self.slider_escala, 1, 1)
        self.lbl_val_escala = QtWidgets.QLabel(f"{self.escala_y:.3f} V")
        painel_controles.addWidget(self.lbl_val_escala, 1, 2)
        
        self.btn_borda = QtWidgets.QPushButton("Subida ↗")
        self.btn_borda.setStyleSheet("background-color: #333333; color: cyan; font-weight: bold; font-size: 14px; padding: 5px; border-radius: 5px;")
        self.btn_borda.clicked.connect(self.alternar_borda)
        painel_controles.addWidget(self.btn_borda, 1, 3)

        painel_controles.addWidget(QtWidgets.QLabel("Posição Y (Elevar):"), 2, 0)
        self.slider_offset = self.criar_slider(0, 4000, int(self.offset_y * 1000), estilo_slider.replace('cyan', 'yellow'))
        self.slider_offset.valueChanged.connect(self.ao_mudar_offset)
        painel_controles.addWidget(self.slider_offset, 2, 1)
        self.lbl_val_offset = QtWidgets.QLabel(f"{self.offset_y:.3f} V")
        painel_controles.addWidget(self.lbl_val_offset, 2, 2)
        
        self.btn_filtro = QtWidgets.QPushButton("Filtro Ruído: OFF")
        self.btn_filtro.setStyleSheet("background-color: darkred; color: white; font-weight: bold; font-size: 14px; padding: 5px; border-radius: 5px;")
        self.btn_filtro.clicked.connect(self.alternar_filtro)
        painel_controles.addWidget(self.btn_filtro, 2, 3)

        painel_controles.addWidget(QtWidgets.QLabel("Nível Trigger (V):"), 3, 0)
        self.slider_trigger = self.criar_slider(0, 4000, int(self.nivel_trigger * 1000), estilo_slider.replace('cyan', 'orange'))
        self.slider_trigger.valueChanged.connect(self.ao_mudar_nivel_trigger)
        painel_controles.addWidget(self.slider_trigger, 3, 1)
        self.lbl_val_trigger = QtWidgets.QLabel(f"{self.nivel_trigger:.3f} V")
        painel_controles.addWidget(self.lbl_val_trigger, 3, 2)
        
        self.btn_media = QtWidgets.QPushButton("Média (16x): OFF")
        self.btn_media.setStyleSheet("background-color: darkred; color: white; font-weight: bold; font-size: 14px; padding: 5px; border-radius: 5px;")
        self.btn_media.clicked.connect(self.alternar_media)
        painel_controles.addWidget(self.btn_media, 3, 3)

        self.btn_daq = QtWidgets.QPushButton("DAQ")
        self.btn_daq.setStyleSheet("background-color: #0055ff; color: white; font-weight: bold; font-size: 15px; padding: 8px; border-radius: 5px; border: 1px solid #00aaff;")
        self.btn_daq.clicked.connect(self.executar_daq)
        painel_controles.addWidget(self.btn_daq, 4, 0, 1, 4)

        layout_principal.addLayout(painel_controles, stretch=1)

    def criar_slider(self, minimo, maximo, atual, estilo):
        slider = QtWidgets.QSlider()
        slider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        slider.setMinimum(minimo)
        slider.setMaximum(maximo)
        slider.setValue(atual)
        slider.setStyleSheet(estilo)
        return slider

    def ao_mudar_tempo(self, val):
        self.tempo_visivel = val / 100.0 
        self.lbl_val_tempo.setText(f"{self.tempo_visivel:.2f} s")
        self.max_pontos = int(self.tempo_visivel * frequencia_amostragem)
        self.y_data_tela = np.zeros(self.max_pontos)
        self.buffer_ondas.clear()
        self.plot_widget.setXRange(0, self.tempo_visivel, padding=0)
        
        PONTOS_PRE = int(self.max_pontos * 0.1)
        tempo_pre = (PONTOS_PRE / frequencia_amostragem)
        self.linha_trigger_v.setPos(tempo_pre)

    def ao_mudar_escala(self, val):
        self.escala_y = val / 1000.0 
        self.lbl_val_escala.setText(f"{self.escala_y:.3f} V")
        self.plot_widget.setYRange(self.offset_y, self.offset_y + self.escala_y, padding=0)

    def ao_mudar_offset(self, val):
        self.offset_y = val / 1000.0 
        self.lbl_val_offset.setText(f"{self.offset_y:.3f} V")
        self.plot_widget.setYRange(self.offset_y, self.offset_y + self.escala_y, padding=0)

    def ao_mudar_nivel_trigger(self, val):
        self.nivel_trigger = val / 1000.0 
        self.lbl_val_trigger.setText(f"{self.nivel_trigger:.3f} V")
        self.linha_trigger_h.setPos(self.nivel_trigger)

    def alternar_trigger(self):
        self.trigger_ativado = not self.trigger_ativado
        self.buffer_ondas.clear() 
        if self.trigger_ativado:
            self.btn_trigger.setText("Trigger: ON")
            self.btn_trigger.setStyleSheet("background-color: darkgreen; color: white; font-weight: bold; font-size: 14px; padding: 5px; border-radius: 5px;")
            self.linha_trigger_h.setVisible(True)
            self.linha_trigger_v.setVisible(True)
        else:
            self.btn_trigger.setText("Trigger: OFF")
            self.btn_trigger.setStyleSheet("background-color: darkred; color: white; font-weight: bold; font-size: 14px; padding: 5px; border-radius: 5px;")
            self.linha_trigger_h.setVisible(False)
            self.linha_trigger_v.setVisible(False)

    def alternar_borda(self):
        if self.borda_trigger == 'subida':
            self.borda_trigger = 'descida'
            self.btn_borda.setText("Descida ↘")
            self.btn_borda.setStyleSheet("background-color: #333333; color: magenta; font-weight: bold; font-size: 14px; padding: 5px; border-radius: 5px;")
        else:
            self.borda_trigger = 'subida'
            self.btn_borda.setText("Subida ↗")
            self.btn_borda.setStyleSheet("background-color: #333333; color: cyan; font-weight: bold; font-size: 14px; padding: 5px; border-radius: 5px;")
        self.buffer_ondas.clear() 

    def alternar_filtro(self):
        self.filtro_ativado = not self.filtro_ativado
        self.buffer_ondas.clear()
        if self.filtro_ativado:
            self.btn_filtro.setText("Filtro Ruído: ON")
            self.btn_filtro.setStyleSheet("background-color: darkgreen; color: white; font-weight: bold; font-size: 14px; padding: 5px; border-radius: 5px;")
        else:
            self.btn_filtro.setText("Filtro Ruído: OFF")
            self.btn_filtro.setStyleSheet("background-color: darkred; color: white; font-weight: bold; font-size: 14px; padding: 5px; border-radius: 5px;")

    def alternar_media(self):
        self.media_ativada = not self.media_ativada
        self.buffer_ondas.clear() 
        if self.media_ativada:
            self.btn_media.setText("Média (16x): ON")
            self.btn_media.setStyleSheet("background-color: darkgreen; color: white; font-weight: bold; font-size: 14px; padding: 5px; border-radius: 5px;")
        else:
            self.btn_media.setText("Média (16x): OFF")
            self.btn_media.setStyleSheet("background-color: darkred; color: white; font-weight: bold; font-size: 14px; padding: 5px; border-radius: 5px;")

    def aplicar_suavizacao(self, array):
        janela = 5
        pesos = np.ones(janela) / janela
        return np.convolve(array, pesos, mode='same')

    def executar_daq(self):
        y_salvar = np.copy(self.y_data_tela)
        x_salvar = np.linspace(0, self.tempo_visivel, len(y_salvar))
        
        # Caminho absoluto solicitado para salvar os arquivos
        diretorio_destino = "/home/rodrigo/Área de Trabalho/LOBI/TCC/python/DAQ/"
        
        # Cria os diretórios caso eles ainda não existam no sistema
        try:
            os.makedirs(diretorio_destino, exist_ok=True)
        except Exception as e:
            print(f"[DAQ] Erro ao verificar/criar diretório de destino: {e}")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_arquivo = f"dados_daq_{timestamp}.csv"
        caminho_completo = os.path.join(diretorio_destino, nome_arquivo)
        
        try:
            with open(caminho_completo, mode='w', newline='') as arquivo:
                escritor = csv.writer(arquivo)
                escritor.writerow(["Tempo (s)", "Tensao (V)"])
                for t, v in zip(x_salvar, y_salvar):
                    escritor.writerow([f"{t:.6f}", f"{v:.4f}"])
            
            print(f"[DAQ] Exportação concluída com sucesso: {caminho_completo}")
        except Exception as e:
            print(f"[DAQ] Erro crítico ao gravar arquivo CSV: {e}")

    def atualizar_grafico(self):
        with buffer_lock:
            arr_bruto = np.array(buffer_interno)
            
        if len(arr_bruto) < self.max_pontos:
            return
            
        arr = self.aplicar_suavizacao(arr_bruto) if self.filtro_ativado else arr_bruto

        atualizar_tela = False
        PONTOS_PRE = int(self.max_pontos * 0.1)
        PONTOS_POS = self.max_pontos - PONTOS_PRE

        if self.trigger_ativado:
            if self.borda_trigger == 'subida':
                condicao = (arr[:-1] < self.nivel_trigger) & (arr[1:] >= self.nivel_trigger)
            else:
                condicao = (arr[:-1] > self.nivel_trigger) & (arr[1:] <= self.nivel_trigger)
                
            indices_trigger = np.where(condicao)[0] + 1
            triggers_validos = [i for i in indices_trigger if (i >= PONTOS_PRE) and ((len(arr) - i - 1) >= PONTOS_POS)]
            
            if triggers_validos:
                idx_trigger = triggers_validos[-1] 
                novo_y = arr[idx_trigger - PONTOS_PRE : idx_trigger + PONTOS_POS]
                
                if len(novo_y) == self.max_pontos:
                    if self.media_ativada:
                        self.buffer_ondas.append(novo_y)
                        self.y_data_tela[:] = np.mean(self.buffer_ondas, axis=0) 
                    else:
                        self.y_data_tela[:] = novo_y
                    atualizar_tela = True
        else:
            novo_y = arr[-self.max_pontos:]
            if len(novo_y) == self.max_pontos:
                if self.media_ativada:
                    self.buffer_ondas.append(novo_y)
                    self.y_data_tela[:] = np.mean(self.buffer_ondas, axis=0)
                else:
                    self.y_data_tela[:] = novo_y
                atualizar_tela = True

        if atualizar_tela:
            x_data = np.linspace(0, self.tempo_visivel, self.max_pontos)
            self.curva_sinal.setData(x=x_data, y=self.y_data_tela)

            tensao_media = np.mean(self.y_data_tela)
            amplitude = np.max(self.y_data_tela) - np.min(self.y_data_tela)
            frequencia_hz = 0.0
            
            if amplitude > 0.050:
                limiar = self.nivel_trigger if self.trigger_ativado else tensao_media
                acima_limiar = self.y_data_tela > limiar 
                mudancas = np.diff(acima_limiar.astype(int))
                
                if not self.trigger_ativado or self.borda_trigger == 'subida':
                    bordas_validas = np.where(mudancas == 1)[0]
                else:
                    bordas_validas = np.where(mudancas == -1)[0]
                    
                if len(bordas_validas) > 1:
                    periodos_pontos = np.diff(bordas_validas)
                    frequencia_hz = frequencia_amostragem / np.mean(periodos_pontos)

            self.label_info.setText(f"Tensão Média (DC): {tensao_media:.3f} V | Frequência Estimada: {frequencia_hz:.1f} Hz")

    def closeEvent(self, event):
        global rodando
        rodando = False
        try:
            ser.close()
        except:
            pass
        event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    janela = OsciloscopioApp()
    janela.show()
    app.exec()