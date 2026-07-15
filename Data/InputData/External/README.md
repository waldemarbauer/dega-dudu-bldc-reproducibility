# External dataset location

The publication repository does not distribute the DUDU-BLDC dataset.

Use:

```bash
uv run python scripts/00_fetch_dataset.py
```

The pinned source is Zenodo record `15522163`, DOI `10.5281/zenodo.15522163`,
archive `DUDU-BLDC.zip`, expected MD5
`b383b9ad1698a3aaf2fc05d4bf48dbe5`.

The fetcher stores and extracts the archive under:

`Data/InputData/External/DUDU-BLDC-v1/`

Do not commit the archive or extracted CSV files.
