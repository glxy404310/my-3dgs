# Official NanoGS frozen baseline: implementation assumptions

Prepared before the real simplification. This is reproduction of an existing
baseline, not a new method, capacity search, or scientific pass gate. The fixed
contract is CONTRACT.md, SHA256
41140ba8c045b81c1ce54512bbc0bf2f671d35125578e215633c099de9a88062.

- Source is saliteta/NanoGS commit cc64a752b5def8d17f9992216bfbb9e533f8364f,
  fetched through the GitHub connector at that exact ref. The actual recursive
  tree contains seven Python files, not eight; no missing utils/__init__.py is
  invented. All 13 author text files, including README, pyproject and LICENSE,
  are byte-verified against their Git blob SHA1 and pinned with SHA256 in
  third_party/NanoGS/<commit>/SOURCE_MANIFEST.json. Only teaser.png is omitted.
  Author bytes stay unchanged; no pip install, patch, replacement algorithm,
  or modification of the existing venv is performed.
- Use the existing /home/administrator/venvs/3dgs/bin/python, Python 3.10.12.
  Runtime NumPy 2.2.6, SciPy 1.15.3 and tqdm 4.70.1 are available. pyproject only
  declares numpy/tqdm, but actual KNN additionally needs scipy.spatial.cKDTree.
  plyfile 1.1.5 is available for independent read-only schema/mapping checks.
  The real author command is python -B -m nanogs.simplification with PYTHONPATH
  set exactly to the pinned src directory; README's old root script path is
  absent in this packaged commit.
- The command explicitly supplies the original defaults: ratio .5, k 16,
  merge_cap .5, opacity_threshold .1, lam_geo 1, lam_sh 1. The unmodified
  CostParams uses n_mc=1, seed=0, eps_cov=1e-8 through its source defaults.
  Original complete KNN edge union, full official rank cost, stable greedy
  disjoint pairing and moment matching remain unchanged. The original
  cKDTree.query(workers=-1) remains unchanged. BLAS/OpenMP environment limits
  are one thread; they do not replace the author's explicit KDTree workers.
- Input is the original 2,050,770-row Truck 30k PLY, SHA256
  3d222d1a0a3e9514730a32bdaf0900ebe8866493081d61d99826d009e9f77736.
  The nominal target is ceil(N0*.5)=1,025,385. Author opacity pruning precedes
  merging and its threshold is min(.1,median(alpha)); the unmodified source
  can in principle stop below target if this first prune already does so.
  The runner records this behavior and requires exactly the nominal target for
  the locked equal-count comparison; a different output count remains UNKNOWN.
  It never pads points, changes thresholds, or calls this a quality result.
- Standard SH3 schema is 62 scalar float32 properties: xyz3, normals3, DC3,
  rest45, opacity1, scale3, quaternion4. Appearance is header-order DC0..2 then
  rest0..44; native rest is channel-major, with R/G/B groups of 15. NanoGS
  treats these as 48 raw appearance fields and mass-averages them when merging.
  Quaternion is wxyz. Position, covariance, alpha and SH all may change.
- Author reader uses sigmoid(logit opacity), exp(clip(logscale,-30,30)) and
  quaternion normalization. Store uses logit with alpha clipped to
  [1e-6,1-1e-6], log(max(scale,1e-12)), and writes normalized quaternion;
  normals are zeroed. These are official behavior, not byte-preserving identity.
  Preflight compares all raw fields/mapping and activated parameters against
  independent plyfile reads, records actual clipping/normalization counts and
  hypothetical store-transform differences, and writes no full identity PLY.
  CPU activation comparison is not native RGB identity certification. An
  independent native identity arm is deferred to later GPU evaluation.
- Synthetic-only witness uses two handmade SH3 rows and checks read activation,
  48-field mapping, wxyz covariance, author moment matching (including SH), and
  a one-row store/read schema. It does not invoke simplify, KNN, rank costs,
  real model reduction, rendering, or assess model quality.
- The real output directory and output.ply must never previously exist. Author
  output is written only there. Input, contract, runner and pinned author hashes
  are checked before/after. stdout/stderr, child actual exit/signal, elapsed time,
  resource observations, output bytes/hash/schema and original failures persist.
  No retry, fallback implementation, parameter change, or smaller model is used.
- The CPU process group has RLIMIT_AS=10GiB. The external wrapper samples
  /proc/<pid>/status RSS every .2 seconds and terminates on RSS>10GiB; sampled
  peak RSS and the kernel child ru_maxrss are separately labeled. RLIMIT_AS is
  a stronger virtual-address constraint, not a claim that RSS reached 10GiB.
  Science stops at the runner's 1790-second total deadline; SIGTERM then at most
  5 seconds before SIGKILL. A root-owned launcher additionally enforces a full
  process limit using timeout TERM at 1795 seconds, kill-after 5 seconds.
  Source/defaults are not altered to fit limits. Total 30-minute feasibility
  and peak memory for the real 2M case remain unknown before execution.

Preparation-only invocations (CPU):

```bash
/home/administrator/venvs/3dgs/bin/python -B \
 '/mnt/d/Documents/my 3dgs/scripts/experiments/run_nanogs_frozen_baseline.py' \
 --mode synthetic --output-dir '/mnt/d/Documents/my 3dgs/research/experiments/nanogs-frozen-baseline-v0/synthetic-witness'

/home/administrator/venvs/3dgs/bin/python -B \
 '/mnt/d/Documents/my 3dgs/scripts/experiments/run_nanogs_frozen_baseline.py' \
 --mode preflight --output-dir '/mnt/d/Documents/my 3dgs/research/experiments/nanogs-frozen-baseline-v0/preflight'
```

Real simplification is root-owned and must wait for independent review:

```bash
bash '/mnt/d/Documents/my 3dgs/research/experiments/launchers/run-nanogs-frozen-baseline.sh'
```

This preparation implements no GPU evaluation or render-speed claim. All prior
#27 and failed/STOP code and results stay unchanged.

Preparation checks executed (CPU, before real simplification): the two-row
synthetic read/activation/wxyz/moment/SH/store witness passed. Full original PLY
preflight read all 2,050,770 rows and all 62 properties; xyz and all 48 SH fields
match independent plyfile mapping exactly. No scale read clipping or store
scale-floor events occurred. There are 185 alpha values above the author's
store upper clip (1-1e-6), none below 1e-6. The effective opacity threshold is
.1 and retains 1,033,349 rows; the original target is 1,025,385. Store-transform
raw differences and independent CPU activation differences are saved in
preflight/PREFLIGHT.json; these are not native RGB identity or quality results.
Input/source hashes before and after both preparation runs match. Neither run
called simplify or imported torch/scene/a renderer. No full identity PLY was
written, and real simplification remains unlaunched.
