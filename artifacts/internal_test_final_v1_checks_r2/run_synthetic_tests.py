from pathlib import Path
import json,os,sys
root=Path.cwd();checks=root/'artifacts/internal_test_final_v1_checks_r2'
sys.path[:0]=[str(root),str(root/'scripts/research'),str(root/'packages/uav_fault')]
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
blocked=[]
def guard(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
    p=Path(os.fsdecode(args[0])).resolve()
    if p.is_relative_to(Path('D:/UAV-Data').resolve()) or p.is_relative_to((root/'artifacts/ens_internal_test_v1_20261105').resolve()):
        blocked.append(str(p));raise PermissionError('Synthetic test attempted real data/model access')
sys.addaudithook(guard)
import pytest
code=pytest.main(['-q','-p','no:cacheprovider','--basetemp='+str(checks/'pytest_tmp_r1'),
 '--junitxml='+str(checks/'pytest_r1.xml'),'tests/test_internal_test_final_v1.py','tests/test_gbt_event_operating_point.py'])
with(checks/'synthetic_access_audit.json').open('x',encoding='utf-8') as f:json.dump({'exit_code':int(code),'real_data_access_attempts_blocked':blocked,'no_test_startup':True},f,indent=2)
raise SystemExit(code)
