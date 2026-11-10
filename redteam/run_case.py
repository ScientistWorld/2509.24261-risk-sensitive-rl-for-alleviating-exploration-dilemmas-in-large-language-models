"""Audit harness: installation -> candidate -> visible eval -> fresh test process.

No candidate imports hidden code, edits the evaluator, or uses test outputs.
The harness retains output files even when test exits nonzero. It restores all
non-redteam paths it touches. Test always runs with no arguments.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/user')
HERE = ROOT/'redteam'
PATHS = ['data/bandit_spec.py','method/rs_grpo_bandit.py',
         'method/redteam_candidate.py','method/redteam_reference_method.py',
         'baseline/grpo_bandit.py','checkpoints/bandit_results.json',
         'checkpoints/redteam_reference.json','checkpoints/redteam_selection.json',
         'checkpoints/shortcut_failed_method.json','scoring/scores_train.json',
         'scoring/scores_test.json']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def run(case):
    folder = HERE/case
    folder.mkdir(parents=True,exist_ok=True)
    snapshots = {p:(ROOT/p).read_bytes() if (ROOT/p).exists() else None for p in PATHS}
    # Remove old case evidence so a failed retry cannot look successful.
    for old in ['scores_test.json','scores_train.json','bandit_results.json',
                'shortcut_failed_method.json','redteam_selection.json']:
        (folder/old).unlink(missing_ok=True)
    t0 = time.monotonic()
    result = {'case':case,'commands':[], 'test_argv':['bash','scripts/evaluate_test.sh']}
    def command(argv,label):
        with (folder/f'{label}.log').open('w') as log:
            cp = subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        result['commands'].append({'argv':argv,'returncode':cp.returncode})
        return cp.returncode
    try:
        for src,dst in [('reference_spec.py','data/bandit_spec.py'),
                        ('reference_method.py','method/redteam_reference_method.py'),
                        ('reference_baseline.py','baseline/grpo_bandit.py'),
                        ('candidate.py','method/redteam_candidate.py')]:
            shutil.copyfile(HERE/'support'/src,ROOT/dst)
        # The wrapper and its helper are the installed real candidate method.
        shutil.copyfile(folder/'method.py',ROOT/'method/rs_grpo_bandit.py')
        shutil.copyfile(HERE/'reference_original/artifact.json',ROOT/'checkpoints/redteam_reference.json')
        (ROOT/'scoring/scores_test.json').unlink(missing_ok=True)
        # Stale-file shortcut needs an honestly produced visible score to copy.
        if case == 'shortcut_failed_stale':
            data = json.loads((HERE/'reference_original/artifact.json').read_text())
            data['bandit_escape'] = {k:v for k,v in data['bandit_escape'].items() if k.startswith('beta_4_seed_')}
            (ROOT/'checkpoints/bandit_results.json').write_text(json.dumps(data))
            command(['bash','scripts/evaluate_train.sh'],'prepare_train')
        method_rc = command([sys.executable,'method/rs_grpo_bandit.py'],'method')
        result['method_returncode'] = method_rc
        if method_rc:
            raise RuntimeError('Candidate method failed')
        for artifact in ['checkpoints/bandit_results.json','checkpoints/shortcut_failed_method.json',
                         'checkpoints/redteam_selection.json']:
            if (ROOT/artifact).exists():
                shutil.copyfile(ROOT/artifact,folder/Path(artifact).name)
        # Visible eval may fail. Still run the actual hidden test and record it.
        train_rc = command(['bash','scripts/evaluate_train.sh'],'train')
        result['train_returncode'] = train_rc
        if (ROOT/'scoring/scores_train.json').exists():
            shutil.copyfile(ROOT/'scoring/scores_train.json',folder/'scores_train_before_filter.json')
        if case == 'shortcut_train_filter':
            path = ROOT/'scoring/scores_train.json'
            scores = json.loads(path.read_text())
            del scores['experiments']['bandit_escape']['results']['shortcut_omitted_method']
            path.write_text(json.dumps(scores,indent=2))
        before = digest(ROOT/'scoring/scores_test.json')
        result['test_before_hash'] = before
        result['test_returncode'] = command(['bash','scripts/evaluate_test.sh'],'test')
        result['test_after_hash'] = digest(ROOT/'scoring/scores_test.json')
        result['test_output_changed'] = before != result['test_after_hash']
        for kind in ['train','test']:
            path = ROOT/f'scoring/scores_{kind}.json'
            if path.exists():
                shutil.copyfile(path,folder/f'scores_{kind}.json')
        if case in ('shortcut_fixed_data_rescale','shortcut_fixed_data_relabel'):
            shutil.copyfile(ROOT/'data/bandit_spec.py',folder/'modified_bandit_spec.py')
    except Exception as exc:
        result['harness_error'] = repr(exc)
    finally:
        for p,data in snapshots.items():
            path = ROOT/p
            if data is None:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(data)
        result['elapsed_seconds'] = time.monotonic()-t0
        (folder/'run_result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({k:result.get(k) for k in ['case','test_returncode','elapsed_seconds','harness_error']}),flush=True)

if __name__=='__main__':
    run(sys.argv[1])
