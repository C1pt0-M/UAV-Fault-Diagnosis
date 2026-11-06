"""Synthetic CPU-only supplement: permanent steps and very-close-logit ties."""
from pathlib import Path
import json
import os
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'packages/uav_fault'), str(ROOT/'scripts/research')]
import torch
import train_causal_mstca_m0 as m0


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    p=m0.load_protocol(ROOT/'configs/experiments/causal_pure_mstca_m0_v1.json')
    model, _=m0.build_model(p,42,ROOT/p['initialization_anchor']['source_path_for_preparation'])
    model.eval()
    result={'device':'cpu','optimizer_steps':0,'trained_checkpoint_used':False,'permanent_steps':[]}
    with torch.no_grad():
        baseline=model(torch.zeros(1,26,150))
        for position,start in [('start',0),('middle',75),('end',140)]:
            x=torch.zeros(1,26,150);x[:,:,start:]=1
            logits=model(x)
            result['permanent_steps'].append({'position':position,'onset_frame_zero_based':start,
                'duration_frames':150-start,'logits':logits[0].tolist(),
                'fault_probability':float(logits.softmax(1)[0,1]),
                'logit_l2_from_zero':float(torch.linalg.vector_norm(logits-baseline))})
        near=torch.tensor([[0.,1e-8],[0.,0.]],dtype=torch.float32)
        result['near_tie']={'logits':near.tolist(),'softmax':near.softmax(1).tolist(),
                            'logit_argmax':near.argmax(1).tolist(),'probability_argmax':near.softmax(1).argmax(1).tolist()}
    path=ROOT/'artifacts/mstca_baseline_audit_v1_20261106/edge_cases.json'
    with path.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
