"""Diagnostic only: val predictions of a saved H1 checkpoint (not used for the verdict)."""
import sys, numpy as np, torch
sys.argv_ = sys.argv
import train_causal_mstca_m0 as m0
source = m0.source
from pathlib import Path
code = Path(sys.argv_[1]); ckpt = sys.argv_[2]; out = sys.argv_[3]
p = m0.load_protocol(code / 'configs/experiments/causal_pure_mstca_m0_v1.json')
ds, frame, _, _ = source.prepare(Path('/root/autodl-tmp/causal_source_v1_20261031/unpacked/data'), p)
from uav_fault.models.cnn.lstm import OnlineMSTCA
cfg = p['training']
model = OnlineMSTCA(in_channels=26, hidden_size=cfg['hidden_size'], num_layers=cfg['num_layers'], num_classes=2, dropout=cfg['dropout']).float()
model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True)); model.cuda()
meta = frame['val']
loader = torch.utils.data.DataLoader(source.IndexedSource(ds['val'], meta.cache_index.to_numpy()), batch_size=512, shuffle=False)
metrics, probs, labels = source.evaluate(model, loader, torch.device('cuda:0'), meta)
np.savez_compressed(out, probabilities=probs, labels=labels, original_indices=meta.original_cache_index.to_numpy(dtype=np.int64))
print('macro_f1', metrics['overall']['macro_f1'])
