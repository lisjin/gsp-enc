# Generalized Shortest-Paths Encoders for AMR-to-Text Generation
**Note:** This branch implements the adjacency matrix multiplication method described in Section 4.2 of the corresponding [paper](https://www.aclweb.org/anthology/2020.coling-main.181.pdf). For Floyd-Warshall, go to the `fl-wa` branch.

**Credit:** This code is based on the [repo](https://github.com/jcyk/gtos) for the 2020 AAAI paper "Graph Transformer for Graph-to-Sequence Learning". We are grateful to the authors for open-sourcing their work.

## Environment Setup
The code is tested with Python 3.6. All dependencies are listed in [requirements.txt](requirements.txt).

## Data Preparation
The instructions to prepare AMR data are given in the [generator_data](./generator_data) folder.

## Pretrained Model
A pretrained checkpoint yielding our paper results can be found [here](https://drive.google.com/drive/folders/1R2gWkRjniEQ1ljBkzwUO6raHn49o7TYE) as the file `batch308999_epoch818`. To verify, execute steps 2--3 below without modifying the script settings. Our model output can be found in the file `batch308999_epoch818_test_out.final` by following the above link.

## Model Training and Evaluation
The following steps should be done in the `generator` folder. The default settings in this repo should reproduce the results in our paper. Please check all scripts for correct arguments before use.

1. Preprocess data and train
    ```
    sh prepare.sh  # vocab and data preprocessing
    sh train.sh
    ```
2. Test and postprocess
    ```
    sh work.sh  # test
    sh test.sh  # postprocess (make sure --output is set)
    ```
3. Evaluate
    ```
    ./multi-bleu.perl  # BLEU eval
    python chrF++.py -H [hyp] -R [ref]  # chrF++ eval
    java -Xmx2G -jar meteor-1.5.jar [hyp] [ref] -l en  # Meteor eval
    ```
