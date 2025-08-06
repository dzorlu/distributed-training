


https://huggingface.co/blog/train_memory

```
nsys profile --output=training_profile.nsys-rep python -m parallelization.fsdp.train --explicit-prefetching --mixed-precision --dcp-api --gpus-per-node 2 --profile
```


Dump memory snapshot, pytorch profling, and nsight dump.

```
rsync -Pavz sky-26ca-dzorlu:/root/sky_workdir/training_profile.nsys-rep logs/
rsync -Pavz sky-26ca-dzorlu:/root/sky_workdir/profiler_logs/rank_1.1754338790131393427.pt.trace.json logs/
rsync -Pavz sky-26ca-dzorlu:/tmp/memory_snapshot_rank0_step6.pkl logs/
```


https://ui.perfetto.dev/#!/viewer?local_cache_key=00000000-0000-0000-e413-45c54e05d0d3
https://docs.pytorch.org/memory_viz

