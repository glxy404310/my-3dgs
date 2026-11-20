"""Read actual one-shot results and freeze source/build identity; no imports of Torch or ops."""
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
FIXED='f62f1cce2cc47a8609c58e4d448ca8eeb6f86e60'
MAIN='e971ea4802908c69eab7e8601bb1eddd2e443754'
GLM='5c46b9c07008ae65cb81ab79cd677ecc1934b903'
def read(name):
 return json.loads((OUT/name).read_text(encoding='utf-8-sig'))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4<<20),b''): h.update(block)
 return h.hexdigest()
target=OUT/'FIXED_BUILD_IDENTITY.json'
assert not target.exists(), 'Identity is immutable once written'
w=read('WATCHDOG.json'); s=read('STAGE_RESULTS.json'); ext=read('EXTERNAL_RUN.json')
i=read('IMPORT_RESULT.json') if (OUT/'IMPORT_RESULT.json').exists() else None
b=read('TRACKED_BEFORE.json') if (OUT/'TRACKED_BEFORE.json').exists() else None
a=read('TRACKED_AFTER.json') if (OUT/'TRACKED_AFTER.json').exists() else None
conditions={
 'outer_exit_zero':ext['exit_code']==0,
 'watchdog_pass':w['decision']=='PRIVATE_BUILD_IMPORT_TOOL_PASS',
 'stage_pass':s['decision']=='PRIVATE_AUTHOR_BUILD_AND_IMPORT_COMPLETE_NO_SCORING',
 'within_900_seconds':w.get('within_900_seconds') is True and ext['wall_seconds']<=900,
 'no_timeout_or_memory_stop':not w.get('timeout') and not w.get('memory_stop'),
 'protected_native_teacher_and_old_fisher_unchanged':w.get('protected_unchanged') is True,
 'all_old_build_artifacts_unchanged':w.get('old_artifacts_unchanged') is True,
 'all_tracked_sources_unchanged':s.get('tracked_sources_unchanged') is True and b==a,
 'no_live_stage_group_processes':w.get('remaining_live_stage_group_pids')==[],
 'exact_official_source_correction':s.get('official_source_correction',{}).get('exact_primary_patch_match') is True,
 'pins_match':bool(b) and tuple(b[x]['HEAD'] for x in ('main','fisher','glm'))==(MAIN,FIXED,GLM),
 'import_pass':bool(i) and i.get('pass') is True,
 'zero_scientific_calls':bool(i) and i.get('scientific_calls')==0,
 'cuda_never_initialized':bool(i) and i.get('cuda_initialized_before_import') is False and i.get('cuda_initialized_after_import') is False,
 'no_scientific_stage_or_install':all(s.get(x) is False for x in ('training','scoring','rendering','model_loaded','install_performed')),
 'tool_sources_unchanged':w.get('script_sha256_before')==w.get('script_sha256_after') and w.get('import_smoke_sha256_before')==w.get('import_smoke_sha256_after'),
 'actual_private_artifact_hashes_match_import':bool(i) and all(sha(i['fisher_paths'][k])==i['fisher_hash'][k] for k in ('python','extension')),
 'all_tracked_blob_hashes_match_index':bool(b) and all(v['git_blob_sha1']==v['actual_git_blob_sha1'] for scope in b.values() for v in scope['tracked'].values() if 'git_blob_sha1' in v),
}
passed=all(conditions.values())
records={p.name:{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(OUT.iterdir()) if p.is_file() and p!=target}
data={
 'schema':'pup-official-fixed-private-build-identity-v1',
 'tool_status':'OFFICIAL_FIXED_PRIVATE_BUILD_IMPORT_PASS' if passed else 'TOOL_PREPARATION_INCONCLUSIVE',
 'pass_conditions':conditions,
 'official_fisher_commit':FIXED,'glm_commit':GLM,'reused_pup_main_commit':MAIN,
 'pup_main_path':'/home/administrator/projects/pup-frozen-baseline-e971ea48',
 'fixed_checkout':'/home/administrator/projects/pup-fisher-official-f62f1cce',
 'private_build_base':'/home/administrator/projects/pup-fisher-official-f62f1cce-private-build-v1',
 'private_lib':s.get('private_build_lib'),
 'tracked_entry_counts':{k:len(v['tracked']) for k,v in b.items()} if b else None,
 'python':{'path':i['fisher_paths']['python'],'sha256':i['fisher_hash']['python']} if i else None,
 'extension':{'path':i['fisher_paths']['extension'],'sha256':i['fisher_hash']['extension']} if i else None,
 'fixed_backward':dict({'path':'/home/administrator/projects/pup-fisher-official-f62f1cce/cuda_rasterizer/backward.cu'},**b['fisher']['tracked']['cuda_rasterizer/backward.cu']) if b else None,
 'execution':{'tool_session':read('TOOL_SESSION.json'),'external':ext,'stage_pid':w.get('stage_pid'),'stage_pgid':w.get('stage_pgid'),'watchdog_pid':w.get('watchdog_pid'),'watchdog_wall_seconds':w.get('external_wall_seconds'),'stage_wall_seconds':s.get('stage_wall_seconds'),'peak_sampled_combined_RSS_bytes':w.get('peak_sampled_combined_RSS_bytes'),'RSS_limit_bytes':w.get('combined_RSS_limit_bytes'),'RSS_poll_seconds':w.get('RSS_poll_seconds'),'internal_TERM_seconds':w.get('active_wall_limit_seconds'),'internal_KILL_delay_seconds':4,'outer_TERM_seconds':895,'outer_KILL_delay_seconds':5,'hard_limit_seconds':900,'remaining_live_stage_group_pids':w.get('remaining_live_stage_group_pids')},
 'native_paths_and_hashes':{'paths':i['native_paths'],'before':i['native_hash_before'],'after':i['native_hash_after']} if i else None,
 'protected_before':w.get('protected_before'),'protected_after':w.get('protected_after'),
 'scientific_calls':i.get('scientific_calls') if i else None,'numerical_verified':False,'environment_ready_certified':False,
 'scope':'Clone, exact official source correction, original-author build, isolated import/link paths and callable binding presence only. No CUDA scientific operator, Scene/model, Fisher scoring, rendering, pruning, training, or installation. Numerical witness and fresh doc-follow execution remain unverified.',
 'records':records,
}
target.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
print(json.dumps({'identity_path':str(target),'identity_sha256':sha(target),'tool_status':data['tool_status'],'failed_conditions':[k for k,v in conditions.items() if not v],'private_lib':data['private_lib'],'python':data['python'],'extension':data['extension'],'fixed_backward':data['fixed_backward'],'execution':data['execution']},indent=2))
raise SystemExit(0 if passed else 2)
