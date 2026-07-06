import sys
from pyqtgraph.Qt import QtWidgets

import daq_serial
from gui import OsciloscopioApp

if __name__ == '__main__':
    # 1. Start the background thread for serial reading
    daq_serial.iniciar_thread_serial()
    
    # 2. Launch the PyQtGraph Application
    app = QtWidgets.QApplication(sys.argv)
    janela = OsciloscopioApp()
    janela.show()
    sys.exit(app.exec())