# PUP private extension: separate pre-effect SH0 and SH3 numerical witnesses

NOT_READY: official-fixed build/import identity is verified, but neither
numerical witness has run at document lock time. This new file preserves the
original SH0 environment doc unchanged. The SH3 challenge was fixed before any
GPU effect, not selected after SH0 outcomes. No Scene instance, Truck, GT,
training, optimizer, installation, source edits or full219 scorer is authorized.

Exact runtime: WSL Ubuntu-22.04, existing
/home/administrator/venvs/3dgs/bin/python (3.10.12), torch2.7.1+cu128.
PYTHONPATH must be exactly
/home/administrator/projects/pup-fisher-official-f62f1cce-private-build-v1/lib.linux-x86_64-3.10:/home/administrator/projects/pup-frozen-baseline-e971ea48.
PUP main e971ea4802908c69eab7e8601bb1eddd2e443754, official fixed Fisher
f62f1cce2cc47a8609c58e4d448ca8eeb6f86e60, GLM5c46b9c07008ae65cb81ab79cd677ecc1934b903.
Actual private wrapper SHA1d4fc43bb67c4552193d340e2d6894ad7ec3c0d04a620fd7655c89ccf855df5b;
_C SHAca99d74bf9d880585b64f75444339bf739276883c8c0f4a85b926fb6e15c9219.
Verified immutable fixed-build record
/mnt/d/Documents/my 3dgs/research/experiments/pup-frozen-baseline-v0/build-official-fixed-v1/FIXED_BUILD_IDENTITY.json
SHA5630aad4281ff1242674910393a085583f0e7f4a62d517e2bf041ad1c3bbe092.
That is build/import PASS only, with numerical_verified=false and
scientific_calls=0. The pre-existing ordinary native extension and old eaef
checkout/build remain untouched. Official PR1 reset fix is existing author
work, not an algorithm modification or novelty claim:
https://github.com/j-alex-hanson/rasterization_and_pup_fisher/pull/1 .

Case1 is the frozen original SH0 contract, KERNEL_CONTRACT.md. Exact schedule
26forwards/1statisticalbackward. Script SHA
5e0694ff2f738177f7d66e6f187042a401993549e0b0b4fdc4871a14023771c7;
contract SHAb7563ff129d970d334bf05c5411b404afe5cdb2d38910346535a7a5c4dbd9495;
original doc SHA049d0f566b1a441401c915544820351d1b389683acc93a14c91a7c02dd41f0af.

Case2 is an independent direction-path and channel-permutation challenge,
KERNEL_CONTRACT_SH3.md, with the same geometry, 4x4 camera, seed0, epsilon1e-3,
FD relativeFrobenius<=.02 and symmetry<=1e-6. It uses active=maxSH3, positive
base DC colors and fixed nonzero three SH1 rows, other12 rows exactly0.
After the original26 schedule, all coefficients' RGB channels are permuted
[1,2,0]: one additional RGB forward maxabs<=2e-5 and one additional official
pool forward/backward with per-G ||Fperm-Fbase||F/||Fbase||F<=1e-4. Total28forwards,
2statisticalbackwards. These are tool identities, not quality gates. No GT,
coefficient fitting or adjusted epsilon/scene/tolerance. Preserve failure.

Each case has its own independent max120seconds including imports/hash/I/O,
1GiB allocated GPU and4GiB combined process-tree RSS: external TERM115/KILL5,
worker calculation105/supervisor110 then up to5s termination. Raw RGB/FD/Gram,
original/permuted parameters, all hashes, call counts, full tracked source,
current GPU/runtime metadata and original partial errors are retained.
Neither actual output directory may exist before its own invocation; no retry.
Fresh agent must preserve tool session exit and external actual host wall.
No case is accepted if host wall exceeds120 even if internal elapsed says PASS.

## Invocation 1: SH0, verbatim, once

After root reviews both frozen preparations, execute this first block only.
The block is byte-for-byte the PowerShell command published in the unchanged
SH0 environment document. Return and inspect actual RESULTS.json,
WATCHDOG.json and EXTERNAL_RUN before considering the second block.
```powershell
$pupWitnessStartedUTC = [DateTime]::UtcNow.ToString('o')
$pupWitnessTimer = [System.Diagnostics.Stopwatch]::StartNew()
& wsl.exe -d Ubuntu-22.04 --exec /usr/bin/env 'PYTHONPATH=/home/administrator/projects/pup-fisher-official-f62f1cce-private-build-v1/lib.linux-x86_64-3.10:/home/administrator/projects/pup-frozen-baseline-e971ea48' 'OMP_NUM_THREADS=1' 'MKL_NUM_THREADS=1' 'OPENBLAS_NUM_THREADS=1' /usr/bin/timeout --signal=TERM --kill-after=5s 115s /home/administrator/venvs/3dgs/bin/python -B -u '/mnt/d/Documents/my 3dgs/scripts/experiments/pup_fisher_kernel_witness.py' --run-once --fixed-build-record '/mnt/d/Documents/my 3dgs/research/experiments/pup-frozen-baseline-v0/build-official-fixed-v1/FIXED_BUILD_IDENTITY.json' --fixed-build-record-sha256 5630aad4281ff1242674910393a085583f0e7f4a62d517e2bf041ad1c3bbe092 2>&1 | Tee-Object -FilePath 'D:\Documents\my 3dgs\research\experiments\pup-frozen-baseline-v0\witness-fixed-v1.EXTERNAL.log'
$pupWitnessExit = $LASTEXITCODE
$pupWitnessTimer.Stop()
$pupWitnessExternal = [ordered]@{started_UTC=$pupWitnessStartedUTC; finished_UTC=[DateTime]::UtcNow.ToString('o'); exit_code=$pupWitnessExit; actual_host_wall_seconds=$pupWitnessTimer.Elapsed.TotalSeconds; hard_limit_seconds=120; within_120_seconds=($pupWitnessTimer.Elapsed.TotalSeconds -le 120); attempts=1; scientific_scope='Fixed 2G/4x4 SH0 kernel tool witness only'}
$pupWitnessExternal | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath 'D:\Documents\my 3dgs\research\experiments\pup-frozen-baseline-v0\witness-fixed-v1.EXTERNAL_RUN.json' -Encoding utf8
if (($pupWitnessExit -ne 0) -or ($pupWitnessTimer.Elapsed.TotalSeconds -gt 120)) { throw 'PUP witness nonzero or outer budget exceeded; preserve original evidence, no retry' }
```

## Mandatory stop between cases

If SH0 exits nonzero, times out, has host wall>120s, lacks complete raw evidence,
has an unknown/integrity state, or does not say
SYNTHETIC_SH0_FISHER_WITNESS_PASS, **stop and do not run invocation2**. Do not
change coefficients, scene, epsilon, tolerances, reset directories or retry.
An actual SH0 PASS does not certify SH3. The cases have separate results.

## Invocation 2: SH3, verbatim, once, only after actual SH0 PASS

The independent wrapper also checks frozen SH0 source identities and its real
PASS/integrity state before any SH3 GPU allocation. Run this separate block
only if invocation1 has passed the preceding complete external/tool checks.
```powershell
$pupWitnessStartedUTC = [DateTime]::UtcNow.ToString('o')
$pupWitnessTimer = [System.Diagnostics.Stopwatch]::StartNew()
& wsl.exe -d Ubuntu-22.04 --exec /usr/bin/env 'PYTHONPATH=/home/administrator/projects/pup-fisher-official-f62f1cce-private-build-v1/lib.linux-x86_64-3.10:/home/administrator/projects/pup-frozen-baseline-e971ea48' 'OMP_NUM_THREADS=1' 'MKL_NUM_THREADS=1' 'OPENBLAS_NUM_THREADS=1' /usr/bin/timeout --signal=TERM --kill-after=5s 115s /home/administrator/venvs/3dgs/bin/python -B -u '/mnt/d/Documents/my 3dgs/scripts/experiments/pup_fisher_kernel_witness_sh3.py' --run-once --fixed-build-record '/mnt/d/Documents/my 3dgs/research/experiments/pup-frozen-baseline-v0/build-official-fixed-v1/FIXED_BUILD_IDENTITY.json' --fixed-build-record-sha256 5630aad4281ff1242674910393a085583f0e7f4a62d517e2bf041ad1c3bbe092 2>&1 | Tee-Object -FilePath 'D:\Documents\my 3dgs\research\experiments\pup-frozen-baseline-v0\witness-fixed-sh3-v1.EXTERNAL.log'
$pupWitnessExit = $LASTEXITCODE
$pupWitnessTimer.Stop()
$pupWitnessExternal = [ordered]@{started_UTC=$pupWitnessStartedUTC; finished_UTC=[DateTime]::UtcNow.ToString('o'); exit_code=$pupWitnessExit; actual_host_wall_seconds=$pupWitnessTimer.Elapsed.TotalSeconds; hard_limit_seconds=120; within_120_seconds=($pupWitnessTimer.Elapsed.TotalSeconds -le 120); attempts=1; scientific_scope='Independent fixed 2G/4x4 SH3 direction and permutation witness only'}
$pupWitnessExternal | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath 'D:\Documents\my 3dgs\research\experiments\pup-frozen-baseline-v0\witness-fixed-sh3-v1.EXTERNAL_RUN.json' -Encoding utf8
if (($pupWitnessExit -ne 0) -or ($pupWitnessTimer.Elapsed.TotalSeconds -gt 120)) { throw 'PUP SH3 witness nonzero or outer budget exceeded; preserve original evidence, no retry' }
```

Root must read both real results before deciding any later action and produce
an immutable certification JSON from actual evidence. These wrappers never
set environment_ready=true or authorize/launch the full PUP scorer. A PASS
only certifies its fixed local synthetic case. No GPU invocation has been
performed during preparation. Allowed preparation checks are py_compile,
stdlib-only --cpu-synthcheck, source/metadata hashes and PowerShell AST parsing.