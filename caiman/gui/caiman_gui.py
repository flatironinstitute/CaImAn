# -*- coding: utf-8 -*-

"""

author: Pengcheng Zhou
email: zhoupc1988@gmail.com
created: 6/16/17
last edited:
"""
import os
from collections import OrderedDict
from PyQt5.QtWidgets import QFileDialog, QWidget, QLabel, QPushButton, \
    QLineEdit, QGridLayout, QApplication

# --------------------------------CLASSES--------------------------------


class FileOpen(QWidget):
    def __init__(self, parent=None, pars=None, directory='.'):
        super(FileOpen, self).__init__(parent)

        if not pars:
            self.file_name = None
            self.dir_folder = os.path.realpath(directory)
            self.name = None
            self.type = None
            self.fr = 10.0
            self.pixel_size = [1]
        else:
            self.file_name = pars.file_name
            if directory != '.':
                self.dir_folder = directory
            else:
                self.dir_folder = pars.dir_folder
            self.name = pars.name
            self.type = pars.type
            self.fr = pars.fr
            self.pixel_size = pars.pixel_size

        self.open_button = QPushButton("Open")
        self.open_button.show()
        self.open_button.clicked.connect(self.load_from_file)

        self.close_button = QPushButton("Close")
        self.close_button.show()
        self.close_button.clicked.connect(self.done)

        # directory
        dir_label = QLabel("Directory")
        self.dir_line = QLineEdit()

        # name
        name_label = QLabel("Name")
        self.name_line = QLineEdit()

        # type
        type_label = QLabel("Type")
        self.type_line = QLineEdit()

        # frame rate
        fs_label = QLabel("Frame rate (Hz)")
        self.fr_line = QLineEdit(str(self.fr))

        # pixel size
        pixel_size_label = QLabel("Pixel size (um)")
        self.pixel_size_line = QLineEdit('1')

        layout = QGridLayout()
        layout.addWidget(self.open_button, 0, 0)
        layout.addWidget(dir_label, 1, 0)
        layout.addWidget(self.dir_line, 1, 1)
        layout.addWidget(name_label)
        layout.addWidget(self.name_line)
        layout.addWidget(type_label)
        layout.addWidget(self.type_line)
        layout.addWidget(fs_label)
        layout.addWidget(self.fr_line)
        layout.addWidget(pixel_size_label)
        layout.addWidget(self.pixel_size_line)
        layout.addWidget(self.close_button)

        self.setLayout(layout)
        self.setWindowTitle("choose video data for processing")

    def load_from_file(self):
        self.file_name, _ = QFileDialog.getOpenFileName(QFileDialog(),
                                                        "open file",
                                                        self.dir_folder)
        self.dir_folder, file_name = os.path.split(self.file_name)
        self.name, self.type = os.path.splitext(file_name)
        self.dir_line.setText(self.dir_folder)
        self.name_line.setText(self.name)
        self.type_line.setText((self.type[1:]))

    def done(self):
        self.fr = float(self.fr_line.text())
        self.pixel_size = [float(i)
                           for i in self.pixel_size_line.text().split(',')]
        self.close()


# -------------------------------FUNCTIONS-------------------------------


def open_file(directory='.'):
    app = QApplication([])
    file_ui = FileOpen(directory=directory)
    file_ui.show()
    # sys.exit(app.exec_())
    app.exec_()
    file_ui.fr = float(file_ui.fr_line.text())
    temp = file_ui.pixel_size_line.text()
    file_ui.pixel_size = [float(i) for i in temp.split(',')]

    file_info = OrderedDict([('file_name', file_ui.file_name),
                             ('dir', file_ui.dir_folder),
                             ('name', file_ui.name),
                             ('type', file_ui.type),
                             ('Fs', file_ui.fr),
                             ('pixel_size', file_ui.pixel_size)])
    return file_info


#----------------------------------RUN----------------------------------
