"""Stage only explicitly scoped B1 text deliverables; never runtime arrays/models."""
from pathlib import Path
import subprocess
r=Path.cwd();c=r/'artifacts/real_v2_b1_learning_curve_checks_resume'
def git(*a):return subprocess.check_output(['git',*a],cwd=r)
assert not [p for p in git('diff','--cached','--name-only','-z').split(b'\0') if p], 'Existing staged changes: stop'
files=[r/p for p in ['scripts/research/real_v2_b1_learning_curve.py','scripts/research/resume_real_v2_b1_learning_curve.py',
 'tests/test_real_v2_b1_learning_curve.py','tests/test_real_v2_b1_resume.py',
 'docs/research/2026-11-08/跨域B1学习曲线结果_v1.md','docs/research/2026-11-08/跨域B1学习曲线_中断现状与补跑评估_v1.md',
 'docs/research/2026-11-08/跨域B1学习曲线_PI指令_补充2.md']]
for folder in ['artifacts/real_v2_b1_learning_curve','artifacts/real_v2_b1_learning_curve_checks','artifacts/real_v2_b1_learning_curve_checks_resume']:
 files.extend(p for p in (r/folder).iterdir() if p.is_file() and p.name not in ['attempt1_real_v2_b1_learning_curve.py'])
manifest=c/'submission_files.txt';manifest.write_text('B1 submission manifest being prepared.\n',encoding='utf-8');files.append(manifest)
files=sorted(set(files))
for p in files:
 assert p.is_file() and p.suffix.lower() not in ['.pkl','.pt','.pth','.npy','.npz','.zip','.gz'], str(p)
 p.read_text(encoding='utf-8')  # Reject accidental binary payloads.
 assert 'pytest_tmp' not in p.parts
subprocess.run(['git','add','-f','--',*[p.relative_to(r).as_posix() for p in files]],check=True,cwd=r)
staged=[s.decode('utf-8') for s in git('diff','--cached','--name-only','-z').split(b'\0') if s]
allowed={p.relative_to(r).as_posix() for p in files};assert set(staged)<=allowed
manifest.write_text('\n'.join(staged)+'\n',encoding='utf-8',newline='\n')
subprocess.run(['git','add','-f','--',str(manifest.relative_to(r))],check=True,cwd=r)
subprocess.run(['git','-c','core.whitespace=trailing-space,space-before-tab,cr-at-eol','diff','--cached','--check'],check=True,cwd=r)
print('EXPLICIT_B1_FILES_STAGED',len(staged),'NO_RUNTIME_BINARY_ARRAY_MODEL')
print(git('diff','--cached','--stat').decode('utf-8'))
