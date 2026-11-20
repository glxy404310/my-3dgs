# Prepared; root review and actual codec proof still required

Only this new evaluation directory was written. `actualrun-v1/` is absent.
No model, scientific PLY, actual RGB array, GPU or renderer was loaded.

- `STATIC_VERIFICATION.json`: stdlib compile/import and 19 old source/JSON
  hash pins pass; no torch/Scene import, zero forwards.
- `SYNTHETIC_CHECK.json`: existing WSL venv, tiny synthetic arrays only. Zero
  MSE/infinite PSNR convention, independent direct 2D Gaussian constant-zero
  population SSIM, camera dtype/shape/hash restoration and tamper rejection,
  and six-column exact crop/quantization pass. No torch/Scene/GPU.
- `bash -n run-evaluation.sh` passes. The launcher's embedded SHA256 list pins
  evaluator, contract, assumptions and19-file old-dependency hash list.

Root should review the complete files once. The only planned scientific launch
is (each SHA is the actual digest obtained from the completed codec attempt):

```sh
bash '/mnt/d/Documents/my 3dgs/research/experiments/direct-sh8-storage-control-v0/evaluation-v1/run-evaluation.sh' \
  '/mnt/d/Documents/my 3dgs/research/experiments/direct-sh8-storage-control-v0/run-v1/RESULT.json' "$RESULT_SHA" \
  '/mnt/d/Documents/my 3dgs/research/experiments/direct-sh8-storage-control-v0/run-v1/SUPERVISOR.json' "$SUPERVISOR_SHA" \
  '/mnt/d/Documents/my 3dgs/research/experiments/direct-sh8-storage-control-v0/run-v1/EXTERNAL.json' "$EXTERNAL_SHA" \
  '/mnt/d/Documents/my 3dgs/research/experiments/direct-sh8-storage-control-v0/run-v1/control.dsh8' "$CODEC_SHA" \
  '/mnt/d/Documents/my 3dgs/research/experiments/direct-sh8-storage-control-v0/run-v1/decoded.ply' "$DECODED_SHA"
```

These shell variable names are interface documentation, not actual hashes or
an executable filled launch command. Root supplies real values only after
successful codec execution and its source/decoded verification.

Main outputs under newly created `actualrun-v1`: CONFIG, CAMERAS plus dtype/shape
restoration proof, RESULTS, SUPERVISOR, EXTERNAL, raw SH8 RGB33, PNG33,
RAW_MANIFEST33, COMBINED_RAW_MANIFEST198, METRICS165, PANEL_MANIFEST10 and
six-column panels10. Old raw files remain at their original pinned locations;
the combined manifest contains absolute paths. The one teacher000016 parity
output is measured and discarded, not substituted into old comparison pixels.

Success remains `SH8_NATIVE_STORAGE_CONTROL_EVALUATED_AUDIT_PENDING`.
Independent CPU-only saved-pixel audit is a separate future step with<=100s
external wall, combined with the main's actual external wall<=300s. No audit
pass is claimed by this preparation. The fresh audit should independently
recompute all165 rows/summaries/rankings/differences,33 source-GT bindings,
198 combined raw hashes,33 new PNGs and10 exact concatenated panels; check
before/after raw-field/model identities,34=1+33 forward declarations,
storage/header/proof bindings, old failure preservation and budgets.

There is no quality gate or method-acceptance conclusion. The32 test views and
the degradation-ranking selection are exploratory. Storage control does not
reduce decoded points, FP32 capacity or imply FPS/VRAM savings.
