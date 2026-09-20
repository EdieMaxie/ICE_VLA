"""Single-robot asynchronous DM05 deployment. Robot API remains unchanged.

All model RPCs have one worker/owner; all TASK_ENV calls stay on the main thread.
25 Hz target, 50-step prediction, separately configured execution windows.
Failure stops issuing actions; hardware stopping/holding remains TASK_ENV's duty.
"""
import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

DEPLOY_VERSION = 'dm05-rtc-pi05-half-speed-v6-gripper-gate-removed'
print(f'[DM05_DEPLOY_LOADED] version={DEPLOY_VERSION} file={__file__}', flush=True)

HZ = 25.0
HORIZON = 50
OVERLAP = 30
COMMIT = 18
EXECUTION_STEPS = int(os.environ.get('DM05_EXECUTION_STEPS', '20'))
PREFETCH_STEPS = 16
# Acceptance gates, not certified physical joint limits. No clipping/retiming.
MAX_JOINT_STEP = 0.05
KEYS = ('left_arm_joint_state', 'left_ee_joint_state',
        'right_arm_joint_state', 'right_ee_joint_state')


def vector(action):
    parts = [np.asarray(action[k], dtype=np.float32).reshape(-1) for k in KEYS]
    if [len(p) for p in parts] != [6, 1, 6, 1]:
        raise ValueError('Unexpected DM05 action layout')
    result = np.concatenate(parts)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite DM05 action')
    return result


def validate_reply(reply, request_id, start, prefix=None):
    if reply.get('arm_speed_scale') != 0.5:
        raise RuntimeError('DM05 arm speed contract mismatch; no action issued')
    if reply['request_id'] != request_id or reply['start_step'] != start:
        raise RuntimeError('RTC response identity mismatch')
    actions = reply['actions']
    if len(actions) != HORIZON:
        raise ValueError('RTC requires full 50-step chunk')
    array = np.stack([vector(a) for a in actions])
    if prefix is not None:
        if int(reply['commit']) != COMMIT:
            raise RuntimeError('RTC commit mismatch')
        if not np.allclose(array[:COMMIT], prefix[:COMMIT], atol=2e-6, rtol=0):
            raise RuntimeError('RTC committed actions changed')
        # Audit entire transition, including the first fully free action.
        joints = [0,1,2,3,4,5,7,8,9,10,11,12]
        # Continuity is a change in velocity, not an absolute speed cap:
        # an already-moving arm must not be forced to brake at a chunk seam.
        prior_velocity=array[COMMIT-1]-array[COMMIT-2]
        seam_velocity=array[COMMIT]-array[COMMIT-1]
        joint = np.abs(seam_velocity-prior_velocity)[joints].max()
        acceleration=np.abs(np.diff(array[COMMIT-2:], n=2, axis=0))
        # Gripper commands retain their original rate; only arm continuity is gated.
        if (joint > MAX_JOINT_STEP+1e-5
                or acceleration[:,joints].max()>0.01201
                or np.abs(np.diff(array[COMMIT-1:,joints],axis=0)).max()>0.12001):
            raise RuntimeError(f'RTC unsafe arm transition: joint={joint:.5f}')
    return actions


def eval_one_episode(TASK_ENV, model_client):
    if not PREFETCH_STEPS < EXECUTION_STEPS < HORIZON-COMMIT:
        raise ValueError('Execution steps must be 17..31 for this 50-step RTC contract')
    print(f'[DM05_RTC_EPISODE] version={DEPLOY_VERSION} execution_steps={EXECUTION_STEPS} arm_speed_scale=0.5 gripper_rate=unchanged', flush=True)
    period = 1.0 / HZ
    request_id = 0
    step = 0
    future = None
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dm05-rpc')

    def rpc(name, obs=None):
        return model_client.call(func_name=name, obs=obs)

    def submit(prefix):
        nonlocal request_id
        # Capture before submitting; never read TASK_ENV in the RPC worker.
        captured = time.monotonic()
        obs = copy.deepcopy(TASK_ENV.get_obs())
        request_id += 1
        obs['rtc'] = {'request_id': request_id, 'start_step': step,
                      'speed_contract': 'pi05-half-speed-v1',
                      'prefix': [] if prefix is None else prefix,
                      'commit': 0 if prefix is None else COMMIT}
        return worker.submit(rpc, 'rtc_infer', obs), request_id, step, captured

    try:
        worker.submit(rpc, 'reset').result()
        caps = worker.submit(rpc, 'rtc_capabilities').result()
        if (caps.get('version') != 1 or caps.get('chunk_size') != HORIZON
                or caps.get('overlap') != OVERLAP or caps.get('commit') != COMMIT
                or caps.get('control_hz') != HZ
                or caps.get('arm_speed_scale') != 0.5
                or caps.get('gripper_retime') != 'delay_runs_keep_rate'):
            raise RuntimeError('RTC server/client mismatch; no robot actions issued')
        if TASK_ENV.is_episode_end():
            return
        future, rid, origin, captured = submit(None)
        queue = validate_reply(future.result(), rid, origin)
        future = None
        queue_start = 0
        switch_step = EXECUTION_STEPS
        ready_queue = None
        due = time.monotonic()
        while not TASK_ENV.is_episode_end():
            now = time.monotonic()
            if now < due:
                time.sleep(due-now)
            if TASK_ENV.is_episode_end():
                break
            if future is not None and future.done():
                reply = future.result()
                elapsed = step-origin
                age = time.monotonic()-captured
                if elapsed >= COMMIT or age >= COMMIT*period:
                    raise RuntimeError(f'RTC late chunk rejected: steps={elapsed}, age={age:.3f}s')
                new_queue = validate_reply(reply, rid, origin, pending_prefix)
                # Logical absolute action index; never execute expired prefix.
                ready_queue = new_queue
                print(f'[DM05 RTC] request={rid} age_ms={age*1000:.1f} '
                      f'dropped={elapsed} model_ms={reply.get("latency_sec",0)*1000:.1f}', flush=True)
                future = None
            if step == switch_step:
                if ready_queue is None:
                    raise RuntimeError('RTC execution deadline missed; no stale replay')
                queue, queue_start = ready_queue, origin
                ready_queue = None
                print(f'[DM05 RTC switch] step={step} prediction_origin={origin} '
                      f'skipped={step-origin}', flush=True)
                switch_step += EXECUTION_STEPS
            offset = step-queue_start
            remaining = len(queue)-offset
            if future is not None and (step-origin >= COMMIT or
                    time.monotonic()-captured >= COMMIT*period):
                raise RuntimeError('RTC deadline missed; stop issuing new actions')
            if remaining <= 0:
                raise RuntimeError('RTC queue exhausted; no stale replay')
            if future is None and ready_queue is None and step == switch_step-PREFETCH_STEPS:
                pending_prefix = np.stack([vector(a) for a in queue[offset:offset+OVERLAP]])
                if not COMMIT < len(pending_prefix) <= OVERLAP:
                    raise RuntimeError('RTC replan window missed')
                future, rid, origin, captured = submit(pending_prefix)
            # Downstream signature, units and action keys are unchanged.
            TASK_ENV.take_action(queue[offset])
            step += 1
            # Never burst catch-up commands after an overrun.
            due = max(due+period, time.monotonic())
    finally:
        # Drain the sole outstanding RPC before another trial can reset client.
        # This does not issue further actions after an end/error.
        worker.shutdown(wait=True, cancel_futures=True)


def eval_one_episode_batch(TASK_ENV, model_client):
    # Existing batch path is unchanged; RTC only applies to the single robot.
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        obs_list = TASK_ENV.get_obs_batch(env_idx_list)
        model_client.call(func_name="update_obs_batch", obs=obs_list)
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)
        chunk_size = len(actions[0])
        for action_idx in range(chunk_size):
            current_action_list = [env_actions[action_idx] for env_actions in actions]
            TASK_ENV.take_action_batch(current_action_list, env_idx_list)
            if TASK_ENV.is_episode_end() or action_idx + 1 == chunk_size:
                break
            running = set(TASK_ENV.get_running_env_idx_list())
            active_batch_idx = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running]
            actions = [actions[i] for i in active_batch_idx]
            env_idx_list = [env_idx_list[i] for i in active_batch_idx]
            model_client.call(func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_idx_list))
