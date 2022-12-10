# Deepfake Detection with Convolutional Neural Networks and Recurrent Neural Networks

## M.Sc. Project

This project used two deepfake datasets: DFDC and CelebDF-v2 and one real faces dataset: Youtube Faces and experimented with two different models: pure CNN - EfficientNet and hybrid CNN-RNN with EfficientNet-GRU.
<br/>
<br/>

### Folder Structure

    models/ - to put pretrained weights of the models
    figures/ - generated for the report
    notebook/ - notebooks used for experimenting
    pso/ - the pso algorithm
    src/ - source code folder
        src/dataset - VideoDataset class and functions to load data, data transformations
        models - classes for Efficientnet and Efficientnet-GRU
        utils - helper functions
        train_val_functions - functions to train and run inference in both models

### Instructions to Run

0.  **Have cuda installed. Install requirements**

    ```
    pip install requirements.txt
    ```

1.  **Download data files and unzip folders into the data folder.**

        1.1. Deepfake detection Dataset parts 0 to 5 in ./data/dfdc:

    https://www.kaggle.com/datasets/phunghieu/deepfake-detection-faces-part-0-0
    https://www.kaggle.com/datasets/phunghieu/deepfake-detection-faces-part-0-1
    https://www.kaggle.com/datasets/phunghieu/deepfake-detection-faces-part-0-2
    https://www.kaggle.com/datasets/phunghieu/deepfake-detection-faces-part-1-0
    https://www.kaggle.com/datasets/phunghieu/deepfake-detection-faces-part-1-1
    https://www.kaggle.com/datasets/phunghieu/deepfake-detection-faces-part-2-0
    https://www.kaggle.com/datasets/phunghieu/deepfake-detection-faces-part-2-1
    https://www.kaggle.com/datasets/phunghieu/deepfake-detection-faces-part-2-2
    https://www.kaggle.com/datasets/phunghieu/deepfake-detection-faces-part-3-0
    https://www.kaggle.com/datasets/phunghieu/deepfake-detection-faces-part-3-1
    https://www.kaggle.com/datasets/phunghieu/deepfake-detection-faces-part-4-0
    https://www.kaggle.com/datasets/phunghieu/deepfake-detection-faces-part-4-1
    https://www.kaggle.com/datasets/phunghieu/deepfake-detection-faces-part-5-0
    https://www.kaggle.com/datasets/phunghieu/deepfake-detection-faces-part-5-1
    https://www.kaggle.com/datasets/phunghieu/deepfake-detection-faces-part-5-2

        1.2. CelebDF-v2 faces extracted straight into ./data/ :

    https://www.kaggle.com/datasets/leovcunha/celebfaces

        1.3. Youtube Faces Extracted in ./data/youtube_faces_dataset :

    https://www.kaggle.com/datasets/leovcunha/youtube-faces-extracted

2.  **The experiments were run in notebooks in the notebook folder. These can be used for testing specific parts.**

    -   notebook/deepfake_detection_data_analysis.ipynb - exploratory data analysis of the datasets
    -   notebook/EfficientnetPure-dup-preliminary.ipynb - prelimiary test with pure Efficientnet Model

    #### **Manual Hyperparameters**:

    -   notebook\manual_hyperparameter_tuning.ipynb - notebook that performs hyperparameter search

    -   notebook/EfficientnetPure-dup-manual_hyperparameters.ipynb - trains the pure Efficientnet Model with hyperparameters found manually
    -   notebook/effnet-gru-manual-hyperpar.ipynb - trains the Efficientnet-GRU Model with hyperparameters found manually

    #### **PSO Hyperparameters**:

    -   notebook/PSO_algorithm_design.ipynb - notebook used to design and test the PSO algorithm
    -   notebook/automatic_hyperparameter_pso-effnetpure.ipynb - hyperparameter search with PSO for efficientnet model
    -   notebook/automatic-hyperparameter-pso-effnetgru.ipynb - hyperparameter search with PSO for efficientnet-gru model
    -   notebook/EfficientnetPure-dup-pso_hyperparameters.ipynb - trains pure Efficientnet with hyperparameters found by PSO.
    -   notebook/effnet-gru-pso-hyperparam.ipynb - trains Efficientnet-GRU with hyperparameters found by PSO.

    #### **Test**:

    -   notebook/test.ipynb - To run tests running inference with the weights of the training models in the datasets used or others.

    Pretrained model weights can be found at:
    https://drive.google.com/drive/folders/1MTatJuHf-Lvelvw2bnBeIcR7metPlTst?usp=share_link
