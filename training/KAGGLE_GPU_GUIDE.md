# Running Training on Kaggle GPU

Yes, this project can train on Kaggle GPU. That should be much faster than your laptop, especially because the model uses Xception transfer learning.

## Recommended Kaggle Steps

1. Open Kaggle and create a new notebook.
2. In notebook settings, set Accelerator to GPU.
3. Add your CT kidney dataset from the right sidebar: Add Input -> Dataset.
4. Upload this project folder, or at minimum upload `training/kaggle_train.py`.
5. Run this in a Kaggle notebook cell:

```bash
!python training/kaggle_train.py
```

The script will try to auto-detect the dataset folder under `/kaggle/input`.

If auto-detection chooses the wrong folder, set the dataset path manually:

```bash
!DATASET_PATH=/kaggle/input/YOUR_DATASET_FOLDER/YOUR_CLASS_PARENT_FOLDER python training/kaggle_train.py
```

The correct `DATASET_PATH` is the folder that directly contains class folders, for example:

```text
/kaggle/input/ct-kidney-dataset-normal-cyst-tumor-stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone
  Cyst/
  Normal/
  Stone/
  Tumor/
```

## Outputs

Kaggle will save trained models here:

```text
/kaggle/working/artifacts/KidneyDiseaseDetection.keras
/kaggle/working/artifacts/KidneyDiseaseDetection.h5
```

After training, download these files from Kaggle output and place them in this local project's `artifacts/` folder.

## Useful Faster Test Run

Before running the full training, test that everything works with fewer epochs:

```bash
!EPOCHS=3 FINE_TUNE_EPOCHS=0 python training/kaggle_train.py
```

## Notes

- Enable Internet in Kaggle notebook settings if TensorFlow needs to download ImageNet weights for Xception.
- The old local notebook used the same folder for train, validation, and test. The Kaggle script uses a real 80/20 train-validation split.
- If Kaggle shows no GPU, check notebook settings again and restart the session.

