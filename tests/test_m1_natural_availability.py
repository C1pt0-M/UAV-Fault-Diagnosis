from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/research'))
from replay_m1_natural_availability import replay_base_only,OnlineDetectionPolicy
from uav_fault.models.cnn.lstm import OnlineMSTCA,RFLY_SEQ_COLUMNS_V2


def test_base_only_M0_has_no_residual_mask_or_filled_residual_channels():
    n=170;grid=1000000+np.arange(n,dtype=np.int64)*20000
    base={'timestamp_us':grid,'canonical_rows':np.ones((n,26)),'row_input_valid':np.ones(n,dtype=bool)}
    norm={'columns':list(RFLY_SEQ_COLUMNS_V2),'mean':[0.]*26,'std':[1.]*26}
    torch.manual_seed(42);model=OnlineMSTCA(in_channels=26);model.eval();before={k:v.clone() for k,v in model.state_dict().items()}
    r=replay_base_only(model,base,norm,OnlineDetectionPolicy(),torch.device('cpu'))
    assert np.flatnonzero(r['available'])[0]==149
    assert r['available'].sum()==21
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert r['events']['policy']['k_on']==3 and r['events']['policy']['onset_th']==.6


def test_invalid_base_frame_is_not_counted_as_normal_prediction():
    n=170;grid=1000000+np.arange(n,dtype=np.int64)*20000
    mask=np.ones(n,dtype=bool);mask[10]=False
    base={'timestamp_us':grid,'canonical_rows':np.ones((n,26)),'row_input_valid':mask}
    norm={'columns':list(RFLY_SEQ_COLUMNS_V2),'mean':[0.]*26,'std':[1.]*26}
    model=OnlineMSTCA(in_channels=26);model.eval();r=replay_base_only(model,base,norm,OnlineDetectionPolicy(),torch.device('cpu'))
    assert not r['available'][:160].any() and r['available'][160:].all()
