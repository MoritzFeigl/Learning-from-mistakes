# Learning from mistakes - Assessing the performance and uncertainty in process-based models

<p align="center">
  <img width="460" src="https://github.com/MoritzFeigl/Learning-from-mistakes/blob/master/learning%20from%20mistakes.png">
</p>

Accompanying code for the publication "Learning from mistakes - Assessing the performance and uncertainty in process-based models"

The code in this repository was used to produce all results in the manuscript. The data is available from https://doi.org/10.5281/zenodo.5185399. 

## Content of the repository
- `main.py` Main python file including the analysis workflow
- `src/` contains the entire code (beside the code in the root directory `main.py` file)
- `model/` contains the XGBoost hyperparameter results and final estimated XGBoost parameters used in this study.  

## Setup to run the code

1. Download the data.zip from https://doi.org/10.5281/zenodo.5185399 and extract it to create the Learning-from-mistakes/data folder with all necessary data files.

2. Install all necessary python packages from requirements.txt.
    with pip:
    Install vi 
    ```
    pip install --user virtualenv
    virtualenv learning_from_mistakes
    pip install -r requirements.txt
    ```
    With conda:
    ```
    conda create --name learning_from_mistakes python=3.8
    conda activate learning_from_mistakes
    conda install --file requirements.txt
    ```

## Run the code
The full study computations can be run from main.py with
```
python main.py
```

## Licence
[Apache License 2.0](https://github.com/MoritzFeigl/Learning-from-mistakes/blob/master/LICENSE)
