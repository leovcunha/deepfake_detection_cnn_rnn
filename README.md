# Deepfake Detection with Convolutional Neural Networks and Recurrent Neural Networks

## M.Sc. Project

This project used two datasets: DFDC and CelebDF-v2 and experimented with two different models: pure CNN - EfficientNet and hybrid CNN-RNN with EfficientNet-GRU.
<br/>
<br/>

### Instructions to Run

1.  Download data files and unzip folders into the data folder.

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

2.  The experiments were run in notebooks in the notebook folder. These can be used for testing specific parts.

    -   notebook/deepfake_detection_data_analysis.ipynb - exploratory data analysis of the datasets
    -   notebook/EfficientnetPure-dup-preliminary.ipynb - prelimiary test with pure Efficientnet Model

    #### **Manual Hyperparameters**:

    -   notebook\manual_hyperparameter_tuning.ipynb - notebook that performs hyperparameter search

    -   notebook/EfficientnetPure-dup-manual_hyperparameters.ipynb - trains the pure Efficientnet Model with hyperparameters found manually

    #### **PSO Hyperparameters**:

    -   notebook/automatic_hyperparameter_pso-effnetpure.ipynb - hyperparameter search with PSO for efficientnet model
    -   notebook/automatic-hyperparameter-pso-effnetgru.ipynb - hyperparameter search with PSO for efficientnet-gru model
    -   notebook/EfficientnetPure-dup-pso_hyperparameters.ipynb - trains pure Efficientnet with hyperparameters found by PSO.

    -   H:\project\notebook\effnet-gru-hyperparam-optimization_pso.ipynb
