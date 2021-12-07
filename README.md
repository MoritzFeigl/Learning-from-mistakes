[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5763568.svg)](https://doi.org/10.5281/zenodo.5763568))
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

2. Create and activate python environment from environment.yml in the project folder with:
  ```
  conda env create --file=environment.yml
  conda activate learning-from-mistakes
  ```

## Run the code
The paper results can be generated from main.py with
```
python main.py
```


## Licence
[Apache License 2.0](https://github.com/MoritzFeigl/Learning-from-mistakes/blob/master/LICENSE)
