# Only change OCC predictor hidden width: D1=256, D2=128.
_base_ = ['./d1-b0-longterm-depth-flashocc.py']
model = dict(occ_head=dict(predictor_hidden_dim=128))
work_dir = './work_dirs/ep3_d2_occ_hidden128'
