import argparse
import torch
p=argparse.ArgumentParser();p.add_argument('checkpoint');a=p.parse_args()
try:
 obj=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
except Exception as e:
 raise RuntimeError('Safe checkpoint load failed. Inspect provenance before attempting weights_only=False') from e
print('Root type:',type(obj).__name__)
if not isinstance(obj,dict): raise TypeError('Expected checkpoint dictionary')
print('Top-level keys:',list(obj)[:30])
for k in ('model','state_dict','model_state_dict'):
 if k in obj and isinstance(obj[k],dict):
  print('Using state dict key:',k);obj=obj[k];break
print('Parameters:',len(obj))
for k,v in list(obj.items())[:35]: print(k,tuple(v.shape) if isinstance(v,torch.Tensor) else type(v).__name__)
