"""DM05-local inference-time RTC with full vector-Jacobian guidance.

Flow time is noise=1 -> actions=0. Guide the predicted clean endpoint x-t*v
towards the overlapping old chunk; no weight updates or upstream source edits.
Formula: PI real-time-chunking / LeRobot RTC, with a quintic soft prefix mask.
"""
from types import MethodType
from types import SimpleNamespace
import numpy as np
import torch
from opendm.model.dm05.dm05_arch import DynamicCache


def prefix_weights(length, commit):
    if not 1 <= commit < length <= 49:
        raise ValueError('Require 1 <= commit < prefix length <= 49')
    w = np.ones(length, dtype=np.float32)
    u = np.linspace(0, 1, length - commit + 1, dtype=np.float32)[1:]
    w[commit:] = 1 - (10*u**3 - 15*u**4 + 6*u**5)
    return w


def install(model):
    model.requires_grad_(False)
    model._dm05_rtc_graphs = {}
    original = model._inference_action_impl

    @torch.no_grad()
    def conditioned(self, input_ids=None, attention_mask=None, pixel_values=None,
                    token_type_ids=None, states=None, image_masks=None,
                    diffusion_steps=10, past_key_values=None, action_mask=None,
                    history_pixel_values=None, history_mask=None, **kwargs):
        context = getattr(self, '_dm05_rtc_context', None)
        if context is None:
            return original(input_ids=input_ids, attention_mask=attention_mask,
                pixel_values=pixel_values, token_type_ids=token_type_ids,
                states=states, image_masks=image_masks, diffusion_steps=diffusion_steps,
                past_key_values=past_key_values, action_mask=action_mask,
                history_pixel_values=history_pixel_values, history_mask=history_mask,
                **kwargs)
        prefix, commit = context
        cache, hidden = self._compute_prefix_cache(input_ids=input_ids,
            attention_mask=attention_mask, pixel_values=pixel_values,
            token_type_ids=token_type_ids, history_pixel_values=history_pixel_values,
            history_mask=history_mask, cache_cls=DynamicCache)
        plen = hidden.shape[1]
        del hidden
        batch = input_ids.shape[0]
        if batch != 1:
            raise ValueError('RTC serves single robot only')
        dtype = self.model.action_in_proj.weight.dtype
        noise = torch.randn(batch, self.model.config.chunk_size,
            self.model.config.action_dim, device=input_ids.device, dtype=dtype)
        x = noise.clone()
        target = torch.zeros_like(x)
        weights = torch.zeros_like(x)
        n, dim = prefix.shape
        target[0, :n, :dim] = torch.as_tensor(prefix, device=x.device, dtype=dtype)
        weights[0, :n, :dim] = torch.as_tensor(
            prefix_weights(n, commit), device=x.device, dtype=dtype)[:, None]
        # State token lengths vary slightly: bucket them like upstream DM05,
        # otherwise new graph capture would stall a moving robot.
        # The adapter token limit is 1024. Use ONE fixed cache profile across
        # the whole episode, not a new capture when state tokens cross a bucket.
        if plen > 1024:
            raise ValueError('RTC observation exceeds the validated 1024-token profile')
        bucket = 1024
        padded_ids = self._pad_suffix_graph_input_ids(input_ids, bucket)
        mask, pos = self._build_suffix_metadata(input_ids=padded_ids,
            prefix_len=bucket, suffix_len=x.shape[1], device=x.device, dtype=dtype)
        result = graph_denoise(self, cache, x, target, weights, mask, pos,
                             action_mask, diffusion_steps)
        # Final proximal projection on the same RTC constraint. The committed
        # prefix is exact; the quintic taper has zero slope at both ends.
        return result * (1-weights) + target * weights

    model._inference_action_impl = MethodType(conditioned, model)


@torch.no_grad()
def graph_denoise(model, cache, noise, target, weights, mask, pos, action_mask, steps):
    keys, values = model._extract_prefix_cache_tensors(cache)
    bucket = mask.shape[-1]-noise.shape[1]
    key = (bucket, tuple(noise.shape), noise.dtype, steps)
    p = model._dm05_rtc_graphs.get(key)
    if p is None:
        if len(model._dm05_rtc_graphs) >= 3:
            # Bounded graph memory. New prompts warm before robot execution.
            model._dm05_rtc_graphs.clear()
        p = SimpleNamespace(state=noise.clone(), target=target.clone(), weights=weights.clone(),
            mask=mask.clone(), pos=pos.clone(),
            keys=tuple(model._make_suffix_graph_cache_tensor(k,bucket) for k in keys),
            values=tuple(model._make_suffix_graph_cache_tensor(v,bucket) for v in values),
            action_mask=torch.ones_like(noise) if action_mask is None else action_mask.clone(),
            time=torch.ones((1,),device=noise.device,dtype=noise.dtype),
            coefficient=torch.ones((),device=noise.device,dtype=noise.dtype))

        def step():
            with torch.enable_grad():
                x = p.state.detach().requires_grad_(True)
                embeds = model._action_input_proj(x * p.action_mask)
                cond = model._build_adarms_cond(p.time)
                out = model.model.action_expert(suffix_embeds=embeds,
                    attention_mask=p.mask, position_ids=p.pos,
                    prefix_cache_keys=p.keys, prefix_cache_values=p.values, adarms_cond=cond)
                velocity = model._action_output_proj(out)
                endpoint = x - p.time[:,None,None]*velocity
                error = ((p.target-endpoint)*p.weights).detach()
                correction = torch.autograd.grad(endpoint,x,error)[0]
            p.state.copy_(x - (velocity-p.coefficient*correction)/steps)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        p.state.copy_(noise)
        p.time.fill_(0.5); p.coefficient.fill_(2.0)
        step()
        eager_output = p.state.clone()
        p.state.copy_(noise)
        p.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(p.graph): step()
        p.state.copy_(noise)
        p.graph.replay()
        # bf16 cancellation near zero makes per-element relative tests brittle.
        # Check aggregate parity plus absolute error, then enforce continuity
        # independently in physical action units after denormalization.
        difference=(p.state-eager_output).float()
        relative=torch.linalg.vector_norm(difference)/torch.linalg.vector_norm(eager_output.float()).clamp_min(1e-6)
        if relative.item()>0.005 or difference.abs().max().item()>0.02:
            raise RuntimeError('RTC graph/eager bf16 parity check failed')
        model._dm05_rtc_graphs[key] = p

    for dst, src in zip(p.keys,keys): model._copy_prefix_cache_tensor(dst,src)
    for dst, src in zip(p.values,values): model._copy_prefix_cache_tensor(dst,src)
    p.mask.copy_(mask); p.pos.copy_(pos); p.state.copy_(noise)
    p.target.copy_(target); p.weights.copy_(weights)
    if action_mask is not None: p.action_mask.copy_(action_mask)
    for i in range(steps):
        t = 1-i/steps
        p.time.fill_(t)
        p.coefficient.fill_(min(10.0,(t*t+(1-t)**2)/max(t*(1-t),1e-8)))
        p.graph.replay()
    return p.state.clone()
