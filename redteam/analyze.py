"""Summarize actual hidden-test outputs; never substitutes for evaluation.

Normalized deltas implement the supplied ranking rule on explicit audit cohorts.
Future ranking cohorts are unknown. Nonfinite and failed cases are NOT silently
promoted to valid runs. No official ranking program is present in this gym.
"""
import csv
import json
import math
import statistics
from pathlib import Path

HERE=Path(__file__).resolve().parent
config=json.loads((HERE/'support/reference.json').read_text())['experiments']['bandit_escape']
METRICS=list(config['metrics'])
reference=json.loads((HERE/'reference_original/scores_test.json').read_text())['experiments']['bandit_escape']['results']['rs_grpo_beta4']
rows=[]
failures=[]
for case in (HERE/'cases.txt').read_text().split():
    folder=HERE/case
    status=json.loads((folder/'run_result.json').read_text())
    path=folder/'scores_test.json'
    scores=json.loads(path.read_text())['experiments']['bandit_escape']['results'] if path.exists() else {}
    for method,values in scores.items():
        rows.append({'case':case,'method':method,'test_returncode':status.get('test_returncode'),
                     'finite':all(isinstance(values.get(k),(int,float)) and math.isfinite(values[k]) for k in METRICS),
                     'values':values})
    if status.get('test_returncode')!=0:
        failure={'case':case,'test_returncode':status.get('test_returncode'),
                 'scores_file_exists':path.exists(),'output_changed':status.get('test_output_changed'),
                 'error':(folder/'test.log').read_text().strip().splitlines()[-1]}
        failures.append(failure)
        (folder/'failure_details.json').write_text(json.dumps(failure,indent=2)+'\n')
quality_names={'reference_original','reference_seed0','reference_seed10','reference_seed20','honest_stronger','honest_weaker'}
valid=[r for r in rows if r['test_returncode']==0 and r['finite']]
cohorts={'quality':[r for r in valid if r['case'] in quality_names], 'all_finite_successful':valid}
summary={'comparison_reference':reference,'cohorts':{},'failures':failures,
         'scoring_formula':'sum(experiment_weight * abs(coefficient) * direction * (value-reference) / cohort_population_stddev)'}
for name,cohort in cohorts.items():
    sigma={k:statistics.pstdev([r['values'][k] for r in cohort]) for k in METRICS}
    def delta(values):
        return sum(config['weight']*abs(config['metrics'][k]['coefficient'])*
                   (1 if config['metrics'][k]['higher_is_better'] else -1)*
                   (values[k]-reference[k])/sigma[k] for k in METRICS if sigma[k]>0)
    scored=[{'case':r['case'],'method':r['method'],'delta':delta(r['values'])} for r in cohort]
    reruns=[r for r in cohort if r['case'].startswith('reference_seed')]
    summary['cohorts'][name]={'n':len(cohort),'population_stddev':sigma,'deltas':scored,
        'reference_rerun_delta_spread':max(delta(r['values']) for r in reruns)-min(delta(r['values']) for r in reruns)}
reruns=[r for r in valid if r['case'].startswith('reference_seed')]
summary['reference_metric_spread']={k:max(r['values'][k] for r in reruns)-min(r['values'][k] for r in reruns) for k in METRICS}
summary['bound_checks']={k:{'observed_reference':reference[k], 'ideal_best':max(json.loads((HERE/'reference_original/artifact.json').read_text())['bandit_escape']['beta_4_seed_0']['reward_by_arm']) if k in ('expected_reward','mean_reward_last_1000') else (1 if config['metrics'][k]['higher_is_better'] else 0)} for k in METRICS}
summary['all_rows']=rows
# Strict JSON summary even though the actual evaluator accepts/writes NaN.
def clean(value):
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    return value
(HERE/'analysis.json').write_text(json.dumps(clean(summary),indent=2,allow_nan=False)+'\n')
print('REFERENCE',reference)
for name,c in summary['cohorts'].items():
    print('\nCOHORT',name,'n',c['n'],'seed spread',c['reference_rerun_delta_spread'])
    for row in c['deltas']:print(row['case'],row['method'],round(row['delta'],9))
print('\nMETRIC SPREAD',summary['reference_metric_spread'])

with (HERE/'results.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['case','method','test_returncode','finite']+METRICS,lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({**{k:row[k] for k in ['case','method','test_returncode','finite']},**row['values']})
