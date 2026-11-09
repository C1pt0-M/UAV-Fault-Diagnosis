from pathlib import Path
import sys,os,json
r=Path.cwd();c=r/'artifacts/real_v2_b1_n2_audit_v1_checks';sys.path[:0]=[str(r),str(r/'packages/uav_fault'),str(r/'scripts/research')];os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1';blocked=[]
def guard(e,args):
 if e!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
 p=Path(os.fsdecode(args[0])).resolve()
 if p.is_relative_to(Path('D:/UAV-Data').resolve()) or p.is_relative_to(Path('D:/RflyMAD').resolve()) or p.suffix.lower() in ('.pkl','.pt','.pth'):
  blocked.append(str(p));raise PermissionError('Only synthetic audit inputs in tests')
sys.addaudithook(guard)
import pytest
attempt=sys.argv[1] if len(sys.argv)>1 else 'r1'
code=pytest.main(['-q','-p','no:cacheprovider','--basetemp='+str(c/f'pytest_tmp_{attempt}'),'--junitxml='+str(c/f'pytest_{attempt}.xml'),'tests/test_real_v2_b1_n2_audit.py'])
(c/f'synthetic_audit_{attempt}.json').write_text(json.dumps({'exit_code':int(code),'forbidden_input_attempts':blocked},indent=2),encoding='utf-8');raise SystemExit(code)
