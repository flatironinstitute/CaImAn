"""
Class to find tif files in directory
"""
import warnings
try:
    from pathlib2 import Path  # py2.7 version
except ModuleNotFoundError:
    from pathlib import Path  # py3 version

try:  # py3
    import tkinter
    from tkinter import filedialog
except ModuleNotFoundError:  # py2.7
    import Tkinter as tkinter
    import tkFileDialog as filedialog


class FileFinder(object):
    def __init__(self, prefix=''):
        self.parent_folder = None
        self.prefix = prefix  # Add a prefix so that the file dialog would open in that folder

    def find_files(self):
        """
        Find all files in a folder
        :return: List of files to parse, first one being the init stack
        """

        path_to_folder = self.update_prefix()
        files_in_folder = self.get_files(path=path_to_folder)
        return files_in_folder

    def update_prefix(self):
        p = Path(self.prefix)
        if not p.exists():
            warnings.warn('Path {} not found, opening a default path.'.format(self.prefix))
            p = Path('.')

        return p

    def get_files(self, path='.'):
        root = tkinter.Tk()
        root.withdraw()
        files = filedialog.askopenfilenames(parent=root, title='Choose files to parse (same FOV)',
                                              filetypes=[('Tif files', '*.tif'), ('Tif files', '*.tiff'),
                                                         ('HDF5 files', '*.h5'), ('HDF5 files', '*.hdf5'),
                                                         ("AVI files", '*.avi'), ('NPY files', '*.npy'),
                                                         ('All files', '*.*')],
                                              initialdir=str(path))
        if len(files) == 0:
            raise UserWarning("No files chosen. Exiting.")

        self.parent_folder = Path(files[0]).parent.absolute()
        return files