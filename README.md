# vmim21cm: paper snapshot

The code, configs and SLURM scripts that produced the results and figures of the paper, as they
were run (comments removed, docstrings shortened to one line; the code itself is unchanged).
Paths are those of the original cluster runs.

## Layout

```
stage1_raw.py           PDF+PS reference: copy the summary table into the run layout
stage1_compress.py      stage 1: train the compressor (MLP / 3D CNN, VMIM / MSE) and export t
merge_export_shards.py  join CNN export shards
stage2_nle.py           stage 2: neural likelihood q(t | theta) (NSF; GMM for the reference line)
stage3_mcmc.py          stage 3: MCMC posteriors for the SBC targets (--batched)
stage4_eval.py          stage 4: SBC calibration, (b, s), posterior volume
make_sbc_list.py        stage-4 list files for a config;  check_status.py: run status
submit_mlp_grid.sh      Path A: stage 1 -> 2 -> 3 chains on SLURM
submit_nle_grid.sh      Path B / reference: stage 1 train -> export -> merge -> 2 -> 3
sbi/                    library (compressors, heads, objectives, cubes, NLE, MCMC, validation)
slurm/                  sbatch scripts called by the submit scripts
configs_paper/          paper configs (below) and the stage-4 lists of the paper table
paper_plot_final/       bs_table.py (run by stage4.sbatch) and the I_c / MI cache builders
AA_plots/scripts/       paper figures (fig_N_*.py), styles in sty/
```

## Paper configs

| arm | stage 1 (compressor) | stage 2-3 (NLE) |
|---|---|---|
| PDF+PS reference | `baseline/baseline_pdf_ps.yaml` (stage1_raw) | `nov19/reference/baseline_pdf_ps_nsf_L8.yaml` (NSF L8h64); GMM line: `nov19/reference/baseline_pdf_ps.yaml` |
| Path A VMIM k=4/8/16 | `patha_pureVMIM3/arm_mlp_vmim_t{4,8,16}_mix2.yaml` | `nov19/patha/L8h64/{vmim_t4_L8,vmim_t8_L8,patha_vmim_t16_L8}.yaml` |
| Path A MSE k=4 | `patha_pureMSE/arm_mlp_mse_t4.yaml` | `nov19/patha/L8h64/mse_t4_L8.yaml` |
| Path B VMIM k=4 | `pathb/vmim/arm_cnn_vmim_{n1,nsqrt2,n2}.yaml` | `nov19/pathb/L8h128/pathb_vmim_{n1,nsqrt2,n2}_L8h128.yaml` |
| Path B VMIM k=8 | `nov19/pathb/compressors/arm_cnn_vmim_{n1,nsqrt2,n2}_t8.yaml` | `nov19/pathb/L8h128/pathb_vmim_{n1,nsqrt2,n2}_t8_L8h128.yaml` |
| Path B MSE k=4 | `pathb/mse/arm_cnn_mse_{n1,nsqrt2,n2}.yaml` | `nov19/pathb/L8h128/pathb_mse_{n1,nsqrt2,n2}_L8h128.yaml` |

n1 / nsqrt2 / n2 = 100 / 50 / 25 h. Seeds: compressor 43, 44, 45 x NLE 42, 43, 44. The NLE configs
reuse the stage-1 summaries of the compressor configs. In the figures, Path B 100 h uses
compressor seeds 43 and 44 only.

Stage 4 of the paper table: `bash configs_paper/submit_s4_paper_table.sh` (lists in
`configs_paper/stage4_lists/`). Figures: `python AA_plots/scripts/fig_N_*.py` (corner and A1_1 via
their sbatch files); the I_c / MI caches come from `paper_plot_final/fig_summary_information.py`,
`patha_mi.py` and `pathb_mi.py`.

Not included: building the PDF+PS summary table and the noise cubes (input data).
