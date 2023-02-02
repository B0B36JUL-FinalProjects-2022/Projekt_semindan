# Intro
[Titanic classification task](https://www.kaggle.com/c/titanic) is a classic Kaggle challenge. This project is my attempt to:
* provide a framework for simple dataframe column operations
* make an sklearn'ish interface for trivial ml models
# Installation
Create a fresh environment (never add anything to your base env) and then
``` julia
add https://github.com/B0B36JUL-FinalProjects-2022/Projekt_semindan
```
The package is imported the following way
```
using Titanic
```
# How to use
This [Jupyter notebook](https://github.com/B0B36JUL-FinalProjects-2022/Projekt_semindan/blob/main/examples/main.ipynb) serves as a guide showcasing all implemented models and some data transformation features.
The data can be downloaded from here: https://www.kaggle.com/c/titanic

# Notes
* Metric enum for K-NN is called with a package prefix, e.g.
    ```
    Titanic.l1
    ```
* Beware, the example notebook is garbage.