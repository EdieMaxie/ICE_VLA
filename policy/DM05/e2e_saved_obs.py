"""Real network + real model, fake environment. NEVER imports robot drivers."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--url',required=True)
p.add_argument('--steps',type=int,default=100)
p.add_argument('--encoded',action='store_true')
args=p.parse_args()
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parents[1]))  # XPolicyLab for client_server imports
sys.path.insert(0,str(root.parents[2]))  # workspace for XPolicyLab imports
from client_server.ws.model_client import WsModelClient
from XPolicyLab.policy.DM05.deploy import eval_one_episode, vector
from XPolicyLab.utils.process_data import unpack_robot_state,get_robot_action_dim_info
data=np.load(root/'rtc_offline_obs.npz')
dims=get_robot_action_dim_info('piper')

class SavedEnv:
    def __init__(self): self.actions=[];self.times=[]
    def is_episode_end(self): return len(self.actions)>=args.steps
    def get_obs(self):
        idx=min((len(self.actions)//25)*25,125)
        vision={cam:{'color':data[f'{idx}_{cam}']} for cam in
                ['cam_head','cam_left_wrist','cam_right_wrist']}
        if args.encoded:
            from XPolicyLab.utils.process_data import encode_image_bit
            for camera in vision.values(): camera['color']=encode_image_bit(camera['color'])
        return {'instruction':'Stand the bottle upright.',
            'state':unpack_robot_state(data[f'{idx}_state'],'joint',dims), 'vision':vision}
    def take_action(self,action):
        self.actions.append(vector(action));self.times.append(time.monotonic())

env=SavedEnv()
failure=None
try:
    with WsModelClient(url=args.url,evaluation_id='dm05-rtc-offline-'+str(time.time_ns()),
                       trial_id='saved-observation-no-robot') as client:
        eval_one_episode(env,client)
except Exception as exc:
    failure=exc
intervals=np.diff(env.times)*1000
report={'steps':len(env.actions),'no_robot':True,'encoded':args.encoded,
        'timestamp_unix':time.time(),'error':None if failure is None else str(failure),
        'interval_ms_p50':float(np.median(intervals)) if len(intervals) else None,
        'interval_ms_p95':float(np.quantile(intervals,.95)) if len(intervals) else None,
        'interval_ms_max':float(intervals.max()) if len(intervals) else None}
(root/('rtc_e2e_encoded.json' if args.encoded else 'rtc_e2e.json')).write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
if failure is not None: raise failure
assert len(env.actions)==args.steps
assert intervals.max()<100, 'Control queue has a >100ms scheduling gap'
