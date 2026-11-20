$ErrorActionPreference='Stop'
$root='D:/Documents/my 3dgs/research/experiments/train-independent-v0'
$s=Get-Content "$root/native/summary.json" -Raw | ConvertFrom-Json
$e=@($s.events)
if(-not $s.complete -or $e.Count -ne 40 -or $s.camera_sequence.Count -ne 4500){throw 'Incomplete run'}
$rows=@($e | ForEach-Object {
 [pscustomobject]@{iteration=$_.iteration;n_before=$_.n_before;n_after=$_.n_after;split_parents=$_.split_parents;pruned=$_.pruned;old_pruned=$_.groups[0].pruned;old_world=$_.groups[0].world;clone_present=$_.groups[1].present;clone_pruned=$_.groups[1].pruned;split_present=$_.groups[2].present;split_pruned=$_.groups[2].pruned;split_world=$_.groups[2].world;split_opacity=$_.groups[2].opacity;split_screen=$_.groups[2].screen}
})
$totals=[ordered]@{}
foreach($k in @('pruned','old_pruned','old_world','clone_present','clone_pruned','split_present','split_pruned','split_world','split_opacity','split_screen')){$totals[$k]=($rows|Measure-Object -Property $k -Sum).Sum}
$out=[ordered]@{scene='train';seed=0;iterations=4500;events=$rows.Count;complete=$s.complete;repo_commit=$s.repo_commit;totals=$totals;events_with_split_world=@($rows|Where-Object split_world -gt 0).Count;fixed_events=@($rows|Where-Object iteration -in @(3600,3700));rows=$rows;scope='Event-local observations in one native baseline. Counts are not unique persistent IDs; overlapping reasons must not be summed; no quality or candidate efficacy inference.'}
$out|ConvertTo-Json -Depth 10|Set-Content "$root/CENSUS_SUMMARY.json" -Encoding utf8
$rows|Export-Csv "$root/census.csv" -NoTypeInformation -Encoding utf8
[pscustomobject]$out|Select-Object scene,events,events_with_split_world,totals,fixed_events|ConvertTo-Json -Depth 6
