## Kaggle Dataset: https://www.kaggle.com/datasets/quora/question-pairs-dataset

**Before Cleaning**

- Rows: 404351
- Columns: 6

- Null Values
    id              0
    qid1            0
    qid2            0
    question1       1
    question2       2
    is_duplicate    0

- Class Distribution:
    is_duplicate
    0   255045
    1   149306
    is_duplicate
    0   0.630752  ~63%
    1   0.369248  ~37%

**After Cleaning**
-Rows: 404348

-NULL Values

    id              0
    qid1            0
    qid2            0
    question1       0
    question2       0
    is_duplicate    0

- Class Distribution

  is_duplicate
  0   255042
  1   149306
  is_duplicate
  0   0.630749  
  1   0.369251

### Dataset Splitting

The cleaned QQP dataset was split into 80% training, 10% validation and 10% test sets using a fixed random seed (SEED = 42)
Validation and test sets were kept clean and untouched for all experiments


### Noise Injection

Uniform label noise was injected into the training set only
For each noise level (10%, 20%, 30%), labels were randomly flipped using a fixed seed (SEED = 42)
Validation and test sets remained clean.

