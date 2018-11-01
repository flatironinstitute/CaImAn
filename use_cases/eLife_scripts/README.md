This folder contains scripts to reproduce the figures appearing [the companion paper](https://www.biorxiv.org/content/early/2018/06/05/339564).

```
@article{giovannucci2018caiman,
  title={CaImAn: An open source tool for scalable Calcium Imaging data Analysis},
  author={Giovannucci, Andrea and Friedrich, Johannes and Gunn, Pat and Kalfon, Jeremie and Koay, Sue Ann and Taxidis, Jiannis and Najafi, Farzaneh and Gauthier, Jeffrey L and Zhou, Pengcheng and Tank, David W and Chklovskii, Dmitri B and Pnevmatikakis, Eftychios A},
  journal={bioRxiv},
  pages={339564},
  year={2018},
  publisher={Cold Spring Harbor Laboratory}
}
```

The scripts `figure_4*, figure_5*, figure_6*` load pre-saved results files for CaImAn batch and CaIman online and reproduce the
corresponding figures. These files are `all_res_web.npz` for CaImAn batch and `all_res_online_web_bk.npz` for CaImAn online,
respectively. To run the algorithms and generate these files you can execute the scripts `./preprocessing_files/Preprocess_CaImAn_batch.py/`
for CaImAn batch and `./preprocessing_files/Preprocess_CaImAn_online.py/` for CaImAn online, respectively. All files and will be made
freely availably soon.
