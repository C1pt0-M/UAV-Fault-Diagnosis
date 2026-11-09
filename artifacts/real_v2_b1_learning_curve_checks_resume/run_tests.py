from pathlib import Path
import os,sys,json
root=Path.cwd();c=root/'artifacts/real_v2_b1_learning_curve_checks_resume';sys.path[:0]=[str(root),str(root/'scripts/research'),str(root/'packages/uav_fault')]
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1';blocked=[]
def guard(event,args):
 if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
 p=Path(os.fsdecode(args[0])).resolve()
 if p.is_relative_to(Path('D:/UAV-Data').resolve()) or p.is_relative_to((root/'artifacts/ens_internal_test_v1_20261105').resolve()):
  blocked.append(str(p));raise PermissionError('Resume tests use synthetic inputs only')
sys.addaudithook(guard)
import pytest
attempt=sys.argv[1] if len(sys.argv)>1 else 'r1'
code=pytest.main(['-q','-p','no:cacheprovider','--basetemp='+str(c/f'pytest_tmp_{attempt}'),'--junitxml='+str(c/f'pytest_{attempt}.xml'),'tests/test_real_v2_b1_resume.py','tests/test_real_v2_b1_learning_curve.py'])
with(c/f'synthetic_audit_{attempt}.json').open('x',encoding='utf-8') as f:json.dump({'exit_code':int(code),'real_data_model_access_attempts':blocked},f,indent=2)
raise SystemExit(code)
