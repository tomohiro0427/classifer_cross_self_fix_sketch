from pathlib import Path
import numpy as np
import torch
from metrics.evaluation_metrics import EMD_CD


# random torch 2048,3 tensor
gen = torch.rand(2048,3)
# copy gen to tgt
tgt = gen.clone()
# random shuffle tgt tensor
# tgt = tgt[torch.randperm(tgt.size(0))]


result = EMD_CD(gen.unsqueeze(0), tgt.unsqueeze(0),1)


print(result)
