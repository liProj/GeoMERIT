"""Fast variant of parallel_controller_fixed.py (2026-11-10, user request: finish within the 1-week reviewer deadline).
Only execution changes: LANES concurrent jobs (two per CPU set) instead of 2, fold-major ordering of the pending
ablations, and reuse of the already valid main report. Data, folds, parameters, thread counts (14) and code are unchanged,
so the frozen protocol signatures still apply."""
LANES=4
from pathlib import Path
import os,sys,json,time,hashlib,subprocess,fcntl,signal
import psutil

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT/'source/geomerit'; WORK=ROOT/'experiment_v1'; RUN=WORK/'runs'
BASE=ROOT/'parallel_execution'; BASE.mkdir(exist_ok=True)
PYTHON=str(ROOT/'.venv/bin/python')
SCOPE=json.loads((BASE/'fixed_scope.json').read_text())
VARIANTS=SCOPE['variants']; MAIN=SCOPE['main']; EXPECTED=10*len(VARIANTS)
SIGNATURE=json.loads((RUN/'protocol.json').read_text())['signature']
FIXED=set(SCOPE['fixed_parameter_variants'])
FIXED_WORK=ROOT/'experiment_fixed_parameters_20261104'
FIXED_SIGNATURE=json.loads((FIXED_WORK/'runs/protocol.json').read_text())['signature']
# 2026-11-10: logs_only runs as a fixed-parameter outer refit too (fixed_logs_only_train.py), in its own result root.
LOGS_FIXED={'logs_only'};LOGS_WORK=ROOT/'experiment_fixed_parameters_logs_only_20261110'
LOGS_SIGNATURE=json.loads((LOGS_WORK/'runs/protocol.json').read_text())['signature']
ALLFIXED=FIXED|LOGS_FIXED
def result_root(v):return LOGS_WORK/'runs' if v in LOGS_FIXED else FIXED_WORK/'runs' if v in FIXED else RUN
def expected_signature(v):return LOGS_SIGNATURE if v in LOGS_FIXED else FIXED_SIGNATURE if v in FIXED else SIGNATURE
lock=(BASE/'controller.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def complete(variant,fold,verify=False):
    dest=result_root(variant)/variant/f'outer_{fold:02d}'; marker=dest/'fold_complete.json'
    if not marker.exists():return False
    data=json.loads(marker.read_text()); prediction=dest/'predictions.parquet'
    assert data['signature']==expected_signature(variant) and data['smoke_only'] is False,str(dest)
    assert prediction.is_file() and prediction.stat().st_size>0,str(dest)
    if variant in ALLFIXED:
        assert data['ablation_protocol']=='fixed_parameters_outer_refit' and data['source_full_fold']==fold
        assert data['parent_signature']==SIGNATURE
        if verify:assert digest(RUN/'full'/f'outer_{fold:02d}'/'selected_parameters.json')==data['source_selection_sha256']
    if verify:assert digest(prediction)==data['prediction_sha256'],str(dest)
    return True

def other_training():
    found=[]
    for p in psutil.process_iter(['pid','cmdline']):
        cmd=p.info['cmdline'] or []
        if p.pid==os.getpid():continue
        if any(str(ROOT) in a for a in cmd) and any(Path(a).name in ['10_revision_train.py','fixed_ablation_train.py','fixed_logs_only_train.py','formal_supervisor.py','11_revision_report.py','essential_report.py','fixed_main_report.py'] for a in cmd):found.append(p.info)
    return found

adopt_path=BASE/'adopt_active.json'
adopt=json.loads(adopt_path.read_text()) if adopt_path.exists() else []
live={p['pid']:p for p in other_training()}
assert set(live).issubset({w['pid'] for w in adopt}), 'Unregistered live training'
for w in adopt:
    assert w['variant'] in VARIANTS
    if w['pid'] in live:
        p=psutil.Process(w['pid'])
        assert abs(p.create_time()-w['created_at'])<0.01 and p.cmdline()==w['actual_command']

for v in VARIANTS:
    for f in range(10):complete(v,f,True)

allcpus=sorted(os.sched_getaffinity(0))
assert len(allcpus)>=20,allcpus
cpu_sets=[allcpus[:10],allcpus[10:20]]
workers=[]
for n in range(LANES):
    w=BASE/f'worker_{n}';w.mkdir(exist_ok=True);r=w/'runs';r.mkdir(exist_ok=True)
    dataset=w/'dataset'
    if not dataset.exists():dataset.symlink_to(WORK/'dataset',target_is_directory=True)
    assert dataset.resolve()==(WORK/'dataset').resolve()
    for v in VARIANTS:
        target=RUN/v;target.mkdir(exist_ok=True)
        link=r/v
        if not link.exists():link.symlink_to(target,target_is_directory=True)
        assert link.resolve()==target.resolve()
    protocol=r/'protocol.json'
    if protocol.exists():assert json.loads(protocol.read_text())['signature']==SIGNATURE
    else:protocol.write_bytes((RUN/'protocol.json').read_bytes())
    fw=BASE/f'fixed_worker_{n}';fw.mkdir(exist_ok=True);fr=fw/'runs';fr.mkdir(exist_ok=True)
    if not (fw/'dataset').exists():(fw/'dataset').symlink_to(WORK/'dataset',target_is_directory=True)
    assert (fw/'dataset').resolve()==(WORK/'dataset').resolve()
    for v in FIXED:
        target=FIXED_WORK/'runs'/v;target.mkdir(exist_ok=True)
        link=fr/v
        if not link.exists():link.symlink_to(target,target_is_directory=True)
        assert link.resolve()==target.resolve()
    fp=fr/'protocol.json'
    if fp.exists():assert json.loads(fp.read_text())['signature']==FIXED_SIGNATURE
    else:fp.write_bytes((FIXED_WORK/'runs/protocol.json').read_bytes())
    lw=BASE/f'logsfixed_worker_{n}';lw.mkdir(exist_ok=True);lr=lw/'runs';lr.mkdir(exist_ok=True)
    if not (lw/'dataset').exists():(lw/'dataset').symlink_to(WORK/'dataset',target_is_directory=True)
    assert (lw/'dataset').resolve()==(WORK/'dataset').resolve()
    for v in LOGS_FIXED:
        target=LOGS_WORK/'runs'/v;target.mkdir(exist_ok=True)
        link=lr/v
        if not link.exists():link.symlink_to(target,target_is_directory=True)
        assert link.resolve()==target.resolve()
    lp=lr/'protocol.json'
    if lp.exists():assert json.loads(lp.read_text())['signature']==LOGS_SIGNATURE
    else:lp.write_bytes((LOGS_WORK/'runs/protocol.json').read_bytes())
    workers.append(dict(id=n,logs_work=str(lw),work=str(w),fixed_work=str(fw),cpus=cpu_sets[n%2]))

state=dict(state='running',pid=os.getpid(),started_at=time.time(),signature=SIGNATURE,
           execution_only=False,fixed_parameter_signature=FIXED_SIGNATURE,result_roots={v:str(result_root(v)) for v in VARIANTS},expected_folds=EXPECTED,scope_id=SCOPE['id'],variants=VARIANTS,workers=workers,events=[],main_report=None)
active={};failed=False;report_process=None;report_file=None;report_done=False

class AdoptedProcess:
    # The original launcher is retired; its still-running children keep their CPU allocation.
    # Their exit codes are unavailable to this process. Completion is explicitly artifact-verified.
    def __init__(self,w): self.w=w; self.pid=w['pid']
    def poll(self):
        try:
            p=psutil.Process(self.pid)
            if abs(p.create_time()-self.w['created_at'])<0.01 and p.status()!=psutil.STATUS_ZOMBIE:
                return None
        except psutil.NoSuchProcess: pass
        return 0 if complete(self.w['variant'],self.w['fold'],True) else 1

for w in adopt:
    n=w['id']; workers[n].update(w); workers[n]['state']='running'
    active[n]=dict(process=AdoptedProcess(w),handle=None,variant=w['variant'],fold=w['fold'],adopted=True)
    state['events'].append(dict(event='adopt_without_restart',worker=n,pid=w['pid'],variant=w['variant'],fold=w['fold']))
    print('ADOPT RUNNING JOB',n,w['variant'],w['fold'],w['pid'],flush=True)
write(BASE/'state.json',state)
if adopt_path.exists():adopt_path.rename(BASE/('adopted_'+str(os.getpid())+'.json'))


def report_valid():
    p=RUN/'report/main_report.json'
    return p.is_file() and json.loads(p.read_text()).get('status')=='complete_main_comparisons' and json.loads(p.read_text()).get('scope_id')==SCOPE['id']

def choose(worker):
    busy={(x['variant'],x['fold']) for x in active.values()}
    for v in MAIN:
        for f in range(10):
            if (v,f) not in busy and not complete(v,f):return v,f
    # Fold-major over the fixed-parameter ablations: every finished round k gives all five ablations on folds 0..k.
    ablations=[v for v in VARIANTS[2:] if v in FIXED]
    for f in range(10):
        for v in ablations:
            if (v,f) not in busy and not complete(v,f) and complete('full',f):return v,f
    for v in VARIANTS[2:]:
        if v in FIXED:continue
        for f in range(10):
            if (v,f) not in busy and not complete(v,f) and complete('full',f):return v,f
    return None

def launch(n,task):
    v,f=task;worker=workers[n]
    filename=BASE/f'worker_{n}_{v}_{f:02d}.log'
    log=filename.open('ab')
    task_work=worker['logs_work'] if v in LOGS_FIXED else worker['fixed_work'] if v in FIXED else worker['work']
    runner=ROOT/'fixed_logs_only_train.py' if v in LOGS_FIXED else ROOT/'fixed_ablation_train.py' if v in FIXED else PROJECT/'scripts/10_revision_train.py'
    command=['taskset','-c',','.join(map(str,worker['cpus'])),PYTHON,'-u',
             str(runner),'--work',task_work,
             '--variants',v,'--folds',str(f)]
    env=dict(os.environ,OMP_NUM_THREADS='14',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
    p=subprocess.Popen(command,cwd=PROJECT,env=env,stdin=subprocess.DEVNULL,
                       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    active[n]=dict(process=p,handle=log,variant=v,fold=f)
    for key in ['actual_command','created_at','returncode','completion_verified','exit_status_source','finished_at']:worker.pop(key,None)
    worker.update(task_work=task_work,created_at=psutil.Process(p.pid).create_time(),pid=p.pid,variant=v,fold=f,state='running',log=str(filename),started_at=time.time(),command=command)
    state['events'].append(dict(event='start',worker=n,variant=v,fold=f,pid=p.pid,time=time.time()))
    print('JOB START',n,v,f,p.pid,flush=True)

if report_valid() and all(complete(v,f) for v in MAIN for f in range(10)):
    report_done=True;state['main_report']=dict(state='complete',reused_existing_valid_report=True)
state['lanes']=LANES

while True:
    for n,item in list(active.items()):
        rc=item['process'].poll()
        if rc is None:continue
        if item['handle'] is not None:item['handle'].close()
        v,f=item['variant'],item['fold']
        valid=rc==0 and complete(v,f,True)
        workers[n].update(state='complete' if valid else 'failed',returncode=None if item.get('adopted') else rc,completion_verified=valid,exit_status_source='artifact_verification_after_adoption' if item.get('adopted') else 'waitpid',finished_at=time.time())
        state['events'].append(dict(event='finish',worker=n,variant=v,fold=f,returncode=None if item.get('adopted') else rc,verified=valid,adopted=item.get('adopted',False),time=time.time()))
        print('JOB END',n,v,f,rc,valid,flush=True)
        del active[n]
        if not valid:failed=True
    main_ready=all(complete(v,f) for v in MAIN for f in range(10))
    if main_ready and report_process is None and not report_done and not failed:
        report_file=(BASE/'main_report.log').open('ab')
        report_process=subprocess.Popen([PYTHON,'-u',str(BASE/'fixed_main_report.py'),'--work',str(WORK)],
                cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=report_file,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2'))
        state['main_report']=dict(state='running',pid=report_process.pid,log=str(BASE/'main_report.log'))
    if report_process is not None and report_process.poll() is not None and not report_done:
        rc=report_process.returncode;report_file.close();report_done=rc==0 and report_valid()
        state['main_report'].update(state='complete' if report_done else 'failed',returncode=rc)
        if not report_done:failed=True
    free=psutil.disk_usage(ROOT).free/1024**3;available=psutil.virtual_memory().available/1024**3
    state.update(updated_at=time.time(),disk_free_gib=free,memory_available_gib=available,
                 completed_folds=sum(complete(v,f) for v in VARIANTS for f in range(10)))
    state['resource_wait']=None
    if not failed:
        for n in range(LANES):
            if n in active:continue
            task=choose(n)
            if task is None:continue
            if free<25 or available<25:
                state['resource_wait']='disk_below_25GiB' if free<25 else 'memory_below_25GiB'
                break
            launch(n,task)
            available-=18
    if failed:
        state['state']='failed_draining' if active else 'failed'
    if state['completed_folds']==EXPECTED and not active and report_done:
        for v in VARIANTS:
            for f in range(10):assert complete(v,f,True)
        state.update(state='complete',returncode=0,completed_at=time.time(),all_predictions_sha256_verified=True)
        write(BASE/'queue_complete.json',state)
        print('PARALLEL QUEUE COMPLETE',flush=True)
    write(BASE/'state.json',state)
    if state['state']=='complete':break
    if failed and not active and (report_process is None or report_process.poll() is not None):raise SystemExit(1)
    time.sleep(15)
