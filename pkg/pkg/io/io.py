import os
from pathlib import Path

import matplotlib.pyplot as plt


def _handle_dirs(pathname, foldername, subfoldername):
    path = Path(pathname)
    if foldername is not None:
        path = path / foldername
        if not os.path.isdir(path):
            os.mkdir(path)
        if subfoldername is not None:
            path = path / subfoldername
            if not os.path.isdir(path):
                os.mkdir(path)
    return path


FIG_PATH = Path(__file__).parent.parent.parent.parent
FIG_PATH = FIG_PATH / "results"
FIG_PATH = FIG_PATH / "figs"


def savefig(
    name,
    format="png",
    dpi=300,
    foldername=None,
    subfoldername=None,
    pathname=FIG_PATH,
    bbox_inches="tight",
    pad_inches=0.5,
    save_on=True,
    transparent=False,
    print_out=False,
    **kws,
):
    if save_on:
        path = _handle_dirs(pathname, foldername, subfoldername)
        savename = path / str(name + "." + format)
        plt.savefig(
            savename,
            format=format,
            facecolor="white",
            transparent=transparent,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            dpi=dpi,
            **kws,
        )
        if print_out:
            print(f"Saved figure to {savename}")


def set_environ():
    os.environ["SAVEFIG_DIR"] = "./maggot_connectome/results"


def set_cwd():
    """Hack for some differences between VS code/jupyter notebook"""
    root_path = "/Users/bpedigo/JHU_code/maggot"
    cwd = os.getcwd()
    if cwd != root_path:
        os.chdir(root_path)
