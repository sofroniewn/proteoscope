if __name__ == "__main__":
    import esm
    import pandas as pd
    import torch
    import zarr
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    GENE2PROTEIN_PATH = "/home/ec2-user/cytoself-data/sequences.csv"
    PROTEIN_EMBED_PATH_F = (
        "/home/ec2-user/cytoself-data/ESM_sequence_embeddings_full.zarr"
    )
    PROTEIN_EMBED_PATH_R = (
        "/home/ec2-user/cytoself-data/ESM_sequence_embeddings_reduced.zarr"
    )

    EMBED_DIM = 1280
    EMBEDDING_LAYER = 33
    TRUNCATION_SEQ_LENGTH = 1024
    TOKENS_PER_BATCH = 2048

    # Load gene2protein mapping
    df = pd.read_csv(GENE2PROTEIN_PATH)
    # sequences = df['Peptide']
    sequences = df["Peptide"].apply(lambda x: x.replace("*", "")).values
    genes = df.index.values
    num_genes = len(genes)

    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()  # disables dropout for deterministic results

    # Create dataset
    dataset = esm.data.FastaBatchedDataset(genes, sequences)
    batches = dataset.get_batch_indices(TOKENS_PER_BATCH, extra_toks_per_seq=1)
    data_loader = DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(TRUNCATION_SEQ_LENGTH),
        batch_sampler=batches,
    )

    z_embedding_prot_f = zarr.open(
        PROTEIN_EMBED_PATH_F,
        mode="w",
        shape=(num_genes, TRUNCATION_SEQ_LENGTH + 1, EMBED_DIM),
        chunks=(1, None, None),
        dtype="float32",
    )

    z_embedding_prot_r = zarr.open(
        PROTEIN_EMBED_PATH_R,
        mode="w",
        shape=(num_genes, 2, EMBED_DIM),
        chunks=(1, None, None),
        dtype="float32",
    )

    if torch.cuda.is_available():
        model = model.to(device="cuda", non_blocking=True)

    for batch_idx, (labels, strs, toks) in tqdm(
        enumerate(data_loader), total=len(data_loader)
    ):
        if torch.cuda.is_available():
            toks = toks.to(device="cuda", non_blocking=True)

        with torch.no_grad():
            out = model(toks, repr_layers=[EMBEDDING_LAYER], return_contacts=False)
            representations = out["representations"][EMBEDDING_LAYER]

        # Save data for each protein
        for i, label in enumerate(labels):
            index = df.index.get_loc(label)
            truncate_len = min(TRUNCATION_SEQ_LENGTH, len(strs[i]))
            bos = representations[i, 0]
            mean = representations[i, 1 : truncate_len + 1].mean(dim=0)
            z_embedding_prot_r[index, 0, :] = bos.detach().cpu().numpy()
            z_embedding_prot_r[index, 1, :] = mean.detach().cpu().numpy()
            z_embedding_prot_f[index, : truncate_len + 1] = (
                representations[i, : truncate_len + 1].detach().cpu().numpy()
            )
