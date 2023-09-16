import os
import pytorch_lightning as pl
import esm
import zarr
from torch.utils.data import DataLoader
import pandas as pd
# from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.callbacks import BasePredictionWriter
# from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
# from fairscale.nn.wrap import enable_wrap, wrap
from dataclasses import dataclass

from torch.distributed.fsdp import MixedPrecision
import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from tqdm import tqdm


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
        if self.model_name == 'esmfold_v1':
            model = esm.pretrained.esmfold_v1()
            model.esm = model.esm.half()
        else:
            model, _ = esm.pretrained.load_model_and_alphabet_hub(self.model_name).half()

        self.model = model.to(self.device)
        self.model.eval()

        # # initialize the model with FSDP wrapper
        # fsdp_params = dict(
        #     mixed_precision=True,
        #     flatten_parameters=True,
        #     state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
        #     cpu_offload=False,  # enable cpu offloading
        # )
        # with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
        #     if self.model_name == 'esmfold_v1':
        #         model = esm.pretrained.esmfold_v1()
        #     else:
        #         model, _ = esm.pretrained.load_model_and_alphabet_hub(self.model_name)
        #     # Wrap each layer in FSDP separately
        #     for name, child in model.named_children():
        #         if name == "layers":
        #             for layer_name, layer in child.named_children():
        #                 wrapped_layer = wrap(layer)
        #                 setattr(child, layer_name, wrapped_layer)
        #     self.model = wrap(model)

    def forward(self, x, embedding_layer):
        self.configure_model()
        if self.model_name == 'esmfold_v1':
            out = self.model.esm(x, repr_layers = range(self.model.esm.num_layers + 1))
            esm_s = torch.stack(
                [v for _, v in sorted(out["representations"].items())], dim=2
            )
            # Drop BOS/EOS
            esm_s = esm_s[:, 1:-1]  # B, L, nLayers,
            esm_s = (self.model.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
            return self.model.esm_s_mlp(esm_s)
        else:
            out = self.model(x, repr_layers = [embedding_layer])
            return out["representations"][embedding_layer]

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        labels, strs, toks = batch
        representations = self(toks, self.embedding_layer)
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
    # import sys
    # import pytorch_lightning
    
    # # Bug fix for openfold
    # # Create a module for the old path
    # sys.modules['pytorch_lightning.utilities.seed'] = sys.modules['lightning_fabric.utilities.seed']


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

    # model_config = ModelConfig(
    #     name="esm2_t36_3B_UR50D",
    #     embedding_layer=34,
    #     embed_dim=2560,
    #     tokens_per_batch=1024,
    #     truncation_seq_length=1024
    # )

    model_config = ModelConfig(
        name="esmfold_v1",
        embedding_layer='all',
        embed_dim=1024,
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
    output_dir = os.path.join(PROTEIN_EMBED_PATH, f'{model_config.name}_{model_config.embedding_layer}')
    pred_writer = CustomWriter(
        output_dir=output_dir,
        write_interval="batch",
    )
    # strategy = FSDPStrategy(
    #     cpu_offload=False, mixed_precision=MixedPrecision(param_dtype=torch.float16)
    # )
    strategy = 'ddp'

    trainer = pl.Trainer(
        devices=4,
        callbacks=[pred_writer],
        accelerator="cuda",
        strategy=strategy,
        precision='16'
    )
    predictions = trainer.predict(
        model, datamodule=data_module, return_predictions=False
    )

    if model.global_rank == 0:
        print('Saving zarr file')
        files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir)])

        sequences_df = sequences_df.iloc[:-1] # drop last seq added to make divisible by 4 !!!!
        num_seq = len(sequences_df)

        z_embedding_prot = zarr.open(
            output_dir + '.zarr',
            mode="w",
                shape=(num_seq, model_config.truncation_seq_length, model_config.embed_dim),
                chunks=(1, None, None),
            dtype="float16",
        )

        if model_config.name == 'esmfold_v1':
            offset = 0
        else:
            # Drop BOS/ EOS 
            offset = 1 

        for file in tqdm(files):
            labels, strs, representations = torch.load(file)
            # Save data for each protein
            for i, label in enumerate(labels):
                index = sequences_df.index.get_loc(label)
                truncate_len = min(model_config.truncation_seq_length, len(strs[i]) - 2 * offset)
                output = representations[i, offset : truncate_len + offset].detach().cpu().numpy()
                z_embedding_prot[index:(index+1), : truncate_len] = output[None, ...]