from pathlib import Path
import os,sys,json
root=Path.cwd();checks=root/'artifacts/real_v2_b1_learning_curve_checks';sys.path[:0]=[str(root),str(root/'scripts/research'),str(root/'packages/uav_fault')]
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1';blocked=[]
template=(root/'artifacts/ens_internal_test_v1_20261105/run_test.py').resolve()
def guard(event,args):
 if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
 p=Path(os.fsdecode(args[0])).resolve()
 if p.is_relative_to(Path('D:/UAV-Data').resolve()) or (p.is_relative_to((root/'artifacts/ens_internal_test_v1_20261105').resolve()) and p!=template):
  blocked.append(str(p));raise PermissionError('Only synthetic tests, no actual data/model access')
sys.addaudithook(guard)
import pytest
attempt=sys.argv[1] if len(sys.argv)>1 else 'r1'
code=pytest.main(['-q','-p','no:cacheprovider','--basetemp='+str(checks/f'pytest_tmp_{attempt}'),'--junitxml='+str(checks/f'pytest_{attempt}.xml'),'tests/test_real_v2_b1_learning_curve.py','tests/test_real_v2_b0_diagnostics.py'])
with(checks/f'synthetic_audit_{attempt}.json').open('x',encoding='utf-8') as f:json.dump({'exit_code':int(code),'real_data_model_access_attempts':blocked},f,indent=2)
raise SystemExit(code)
