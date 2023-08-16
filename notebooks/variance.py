import dask.array as da
import numpy as np
import pandas as pd


def get_dataset_var(dataset):
    images_da = da.from_zarr(dataset.images)[dataset.labels.index.values]
    return np.asarray(images_da.var())


# pdm = ProteoscopeDataModule(
#     images_path=config.data.images_path,
#     labels_path=config.data.labels_path,
#     batch_size=config.trainer.batch_size,
#     num_workers=config.trainer.num_workers,
# )
# pdm.setup()

train_var = get_dataset_var(pdm.train_dataset)
val_images_var = get_dataset_var(pdm.val_images_dataset)
val_proteins_var = get_dataset_var(pdm.val_proteins_dataset)

vars = pd.DataFrame(
    {
        "all": [var],
        "train_dataset": [train_var],
        "val_images_dataset": [val_images_var],
        "val_proteins_dataset": [val_proteins_var],
    }
)

datapath = "/home/ec2-user/cytoself-data/variance.csv"
vars.to_csv(datapath)
