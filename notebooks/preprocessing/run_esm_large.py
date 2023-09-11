import os
import pytorch_lightning as pl
import esm
import zarr
from torch.utils.data import DataLoader
import pandas as pd
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.callbacks import BasePredictionWriter
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap
from dataclasses import dataclass

from torch.distributed.fsdp import MixedPrecision
import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler


@dataclass
class ModelConfig:
    name: str
    embedding_layer: int
    embed_dim: int
    tokens_per_batch: int
    truncation_seq_length: int


class ESMModel(pl.LightningModule):
    def __init__(self, gene_to_protein, model_config):
        super(ESMModel, self).__init__()
        self.gene_to_protein = gene_to_protein
        self.num_genes = len(gene_to_protein)
        self.embedding_layer = model_config.embedding_layer  # 36
        self.truncation_seq_length = model_config.truncation_seq_length  # 1024
        self.tokens_per_batch = model_config.tokens_per_batch  # 1024
        self.embed_dim = model_config.embed_dim  # 2560
        self.model_name = model_config.name
        self.model = None

    def configure_model(self):
        if self.model is not None:
            return

        # initialize the model with FSDP wrapper
        fsdp_params = dict(
            mixed_precision=True,
            flatten_parameters=True,
            state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
            cpu_offload=False,  # enable cpu offloading
        )
        with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
            model, _ = esm.pretrained.load_model_and_alphabet_hub(self.model_name)
            # Wrap each layer in FSDP separately
            for name, child in model.named_children():
                if name == "layers":
                    for layer_name, layer in child.named_children():
                        wrapped_layer = wrap(layer)
                        setattr(child, layer_name, wrapped_layer)
            self.model = wrap(model)

    def forward(self, x, **kwargs):
        self.configure_model()
        return self.model(x, **kwargs)

    def predict_step(self, batch, batch_idx):
        labels, strs, toks = batch
        out = model(toks, repr_layers=[self.embedding_layer], return_contacts=False)
        representations = out["representations"][self.embedding_layer]
        return labels, strs, representations

    def configure_optimizers(self):
        pass


class ESMDataModule(pl.LightningDataModule):
    def __init__(self, sequences_df, model_config):
        super(ESMDataModule, self).__init__()
        # sequences = gene_to_protein['Peptide']
        sequences = (
            sequences_df["Peptide"].apply(lambda x: x.replace("*", "")).values
        )
        index = sequences_df.index.values
        self.num_sequences = len(index)

        self.index = index
        self.sequences = sequences
        self.alphabet = None
        self.tokens_per_batch = model_config.tokens_per_batch
        self.truncation_seq_length = model_config.truncation_seq_length

    def predict_dataloader(self):
        dataset = esm.data.FastaBatchedDataset(self.index, self.sequences)
        batches = dataset.get_batch_indices(self.tokens_per_batch, extra_toks_per_seq=1)

        # Use batches list directly with BatchSampler
        batch_sampler = BatchSampler(batches, batch_size=1, drop_last=True)
        return DataLoader(
            dataset,
            shuffle=False,
            num_workers=0,
            collate_fn=self.alphabet.get_batch_converter(self.truncation_seq_length),
            batch_sampler=batch_sampler,
        )


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(
            prediction,
            os.path.join(
                self.output_dir, f"predictions_{trainer.global_rank}_{batch_idx}.pt"
            ),
        )


if __name__ == "__main__":
    GENE2PROTEIN_PATH = "/home/ec2-user/cytoself-data/sequences.csv"
    PROTEIN_EMBED_PATH = "/home/ec2-user/cytoself-data/"

    # SEQ_LOC_PATH = "/home/ec2-user/esm-data/protein_loc.csv"
    # PROTEIN_EMBED_PATH = "/home/ec2-user/esm-data/"

    # model_config = ModelConfig(
    #     name="esm2_t33_650M_UR50D",
    #     embedding_layer=32,
    #     embed_dim=1280,
    #     tokens_per_batch=1024,
    #     truncation_seq_length=1024
    # )

    model_config = ModelConfig(
        name="esm2_t36_3B_UR50D",
        embedding_layer=34,
        embed_dim=2560,
        tokens_per_batch=1024,
        truncation_seq_length=1024
    )

    # model_config = ModelConfig(
    #     name="esm2_t48_15B_UR50D",
    #     embedding_layer=48,
    #     embed_dim=5120,
    #     tokens_per_batch=512,
    #     truncation_seq_length=1024
    # )

    # Initialize DataModule and Model
    sequences_df = pd.read_csv(GENE2PROTEIN_PATH)
    sequences_df = sequences_df.append(sequences_df.iloc[-1]) # make divisible by 4 !!!!

    # sequences_df = pd.read_csv(SEQ_LOC_PATH)
    data_module = ESMDataModule(sequences_df, model_config)
    data_module.alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
    model = ESMModel(sequences_df, model_config)

    # Use the Trainer for prediction
    pred_writer = CustomWriter(
        output_dir=os.path.join(PROTEIN_EMBED_PATH, f'{model_config.name}_{model_config.embedding_layer}'),
        write_interval="batch",
    )
    strategy = FSDPStrategy(
        cpu_offload=False, mixed_precision=MixedPrecision(param_dtype=torch.float16)
    )

    trainer = pl.Trainer(
        devices=4,
        callbacks=[pred_writer],
        accelerator="cuda",
        strategy=strategy,
    )
    predictions = trainer.predict(
        model, datamodule=data_module, return_predictions=False
    )
