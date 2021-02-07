jupytext --to notebook --output maggot_connectome/docs/$1.ipynb maggot_connectome/scripts/$1.py
jupyter nbconvert --to notebook --stdout --execute --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.record_timing=True maggot_connectome/docs/$1.ipynb 
# {'metadata': {'path': run_path}}
# https://github.com/jupyter/nbconvert/blob/7ee82983a580464b0f07c68e35efbd5a0175ff4e/nbconvert/preprocessors/execute.py#L63
