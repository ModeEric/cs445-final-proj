# cs445-final-proj

Github Link: https://github.com/ModeEric/cs445-final-proj

Data link: https://drive.google.com/drive/folders/1xRSwGkX3MyESLVx36whvVmQdcbsWUZsB


TrackNet best model: model_best1.pth

TrackNet VGG with ECA: model_best_eca1.pth

TrackNet Both with ECA:model_best_eca2.pth



To run this project:
1. Download the desired dataset from the data link above-- "Dataset" for the tennis dataset, "padel_dataset" for the padel dataset
2. Locate the TrackNet.ipynb file (or TrackNet_padel.ipynb)
3. If only testing, import the desired model's filepath under the Test section. For example, if you want to use the eca1 model, you should have the line "model_save_name = './Result/model_best_eca1.pth'" (depending on where the file is).
4. Make sure that the data is downloaded and extracted, and locate the filepath. Put this filepath in the Data section as the base_path. For example, if you are using the padel dataset and it is located in './padel_dataset', you should have the line "base_path = './padel_dataset'"
