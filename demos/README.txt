The demos serve a few purposes:
1) They demonstrate use of the Caiman software with some sample data either bundled or automatically fetchable
2) They act as a template for you to write your own analysis
3) They can be directly used if they already do almost exactly what you need to do with Caiman.

There are two types of demos, each in their own folder:
* Jupyter demos are meant to be run interactively, in Jupyter or potentially one of its variants or something else compatible with Jupyter notebooks. They run in a cell-oriented way, allowing you to step through each part of the Caiman pipeline, seeing results and possibly changing things (values of parameters, files to work on, similar) between them.
* CLI demos are meant to be run from the commandline of your operating system (Powershell or CMD on Windows, Bash/Zsh inside a Terminal on Linux or OSX). When running Caiman this way, you might or might not want to see visualisations, and the arguments you provide to these demos control their behaviour.

The CLI demos used to have a different intended use case, also being meant for cell-based interactive use, but in an IPython-based IDE like Spyder; this usage is no longer supported (use something compatible with Jupyter notebook formats for that). The earlier form of the demos is available in our use_cases repo in this directory:
https://github.com/flatironinstitute/caiman_use_cases/tree/main/use_cases/old_cli_demos

