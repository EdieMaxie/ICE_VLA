"""Inference-only bounded-derivative projection; never modifies committed actions.

This is an additional deployment constraint, not the RTC paper's sampler.
Thresholds are acceptance settings, not hardware-certified safety limits.
Large changes fail closed rather than disguising a different model strategy.
"""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from threadpoolctl import threadpool_limits

@threadpool_limits.wrap(limits=1, user_api='blas')
def project(chunk, prefix, commit):
    out=np.asarray(chunk,dtype=np.float32).copy()
    source=out.copy()
    out[:commit]=prefix[:commit]
    # Preserve ordinary reaching velocity; enforce a tighter seam gate below.
    # A small cap over the entire future chunk would silently slow the policy.
    vmax=np.full(14,0.12,dtype=np.float32); vmax[[6,13]]=0.15
    amax=np.full(14,0.012,dtype=np.float32); amax[[6,13]]=0.05
    velocity=out[commit-1]-out[commit-2]
    # Already-issued prefix must be left untouched, even if its velocity is
    # outside these gates. Do not introduce an abrupt braking target.
    if np.any(np.abs(velocity)>vmax+1e-6):
        raise RuntimeError('RTC prefix velocity exceeds continuity envelope')
    # Solve for the WHOLE uncommitted suffix: greedy acceleration clamping can
    # overshoot because it cannot anticipate braking. These are convex QPs.
    n=len(out)-commit
    d1=np.eye(n)-np.eye(n,k=-1)
    d2=np.eye(n)-2*np.eye(n,k=-1)+np.eye(n,k=-2)
    matrix=np.concatenate([d1,d2])
    for j in range(14):
        first=np.zeros(n); first[0]=-out[commit-1,j]
        second=np.zeros(n)
        second[0]=-2*out[commit-1,j]+out[commit-2,j]
        second[1]=out[commit-1,j]
        shift=np.concatenate([first,second])
        bound=np.concatenate([np.full(n,vmax[j]),np.full(n,amax[j])])
        target=source[commit:,j].astype(np.float64)
        problem=LinearConstraint(matrix,-bound-shift,bound-shift)
        feasible=out[commit-1,j]+np.arange(1,n+1)*velocity[j]
        fit=minimize(lambda z: .5*np.sum((z-target)**2), feasible,
            jac=lambda z:z-target, constraints=[problem], method='SLSQP',
            options={'ftol':1e-10,'maxiter':100})
        if not fit.success or np.max(np.abs(matrix@fit.x+shift)-bound)>1e-5:
            raise RuntimeError(f'RTC continuity QP failed: {fit.message}')
        out[commit:,j]=fit.x
    error=np.abs(out[commit:]-source[commit:])
    joint_error=float(error[:,[0,1,2,3,4,5,7,8,9,10,11,12]].max())
    gripper_error=float(error[:,[6,13]].max())
    if joint_error>0.15 or gripper_error>0.3:
        raise RuntimeError(f'RTC strategy disagreement too large: joints={joint_error:.4f}, gripper={gripper_error:.4f}')
    return out, {'joint_projection_max':joint_error,'gripper_projection_max':gripper_error}
