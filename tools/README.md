# Release Tools (maintainers only)

These tools build the CaptionStew caption release on Hugging Face
([`RayTseng0406/CaptionStew`](https://huggingface.co/datasets/RayTseng0406/CaptionStew)).
Users of the data-prep recipes do not need them.

## Building and publishing v2

v1 stored absolute paths from the authors' cluster. `build_release_v2.py` rewrites every manifest to use
source-relative path keys that `link_captions.py` can resolve; see its docstring for the full list of changes.

1. Tag the current release so that v1 stays reachable (once):

       hf repos tag create RayTseng0406/CaptionStew v1.0 --repo-type dataset -m "CaptionStew v1 (absolute paths)"

2. Download v1:

       hf download RayTseng0406/CaptionStew --repo-type dataset --revision v1.0 --local-dir captionstew_v1

3. Build v2 (about 20 minutes and under 5 GB of RAM in total with 8 workers):

       python tools/build_release_v2.py captionstew_v1 captionstew_v2 --workers 8

   This writes the manifests, `stats.json` and the dataset card (`README.md`, filled from `dataset_card.md`).

4. Upload v2 to `main`. Subset shards are renamed (`shard_0_corrected.jsonl.gz` -> `shard_0.jsonl.gz`,
   `400k/400k_CaptionStew_400k_30s_corrected.jsonl.gz` -> `400k/shard_0.jsonl.gz`), so delete the old names in the same commit:

       hf upload RayTseng0406/CaptionStew captionstew_v2 . --repo-type dataset \
         --delete "*_corrected.jsonl.gz" \
         --commit-message "CaptionStew v2: path keys, merged duplicate cuts, dataset card"

5. Tag v2:

       hf repos tag create RayTseng0406/CaptionStew v2.0 --repo-type dataset -m "CaptionStew v2 (path keys)"
