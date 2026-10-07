"""Rápido acoustic model: Brazilian Portuguese letters in, 64-dim latents at 25 Hz out (flow matching)."""
import torch
import torch.nn as nn

from ._dit import Acoustic, rope


def word_layout(text):
    """Letter -> word index and the first letter of that word (a word owns its trailing space)."""
    starts = [i for i, ch in enumerate(text) if ch != " " and (i == 0 or text[i - 1] == " ")] or [0]
    bounds = [0] + starts[1:] + [len(text)]
    cw, wstart = [], []
    for w, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        cw += [w] * (b - a)
        wstart += [a] * (b - a)
    return cw, wstart


def frame_layout(counts):
    """Word frame counts -> each frame's word and its position inside that word."""
    fw = torch.repeat_interleave(torch.arange(len(counts)), counts)
    start = (counts.cumsum(0) - counts)[fw]
    fp = ((torch.arange(len(fw)) - start).double() / counts[fw].double()).float()
    return fw, fp


class RapidoModel(nn.Module):
    def __init__(self, cfg, vocab, n_spk):
        super().__init__()
        self.cfg = cfg
        self.net = Acoustic(cfg, vocab, cfg.get("shared_ada", True))
        self.spk = nn.Embedding(n_spk, cfg["d"])
        self.register_buffer("lat_mean", torch.zeros(cfg["latent_dim"]))
        self.register_buffer("lat_std", torch.ones(cfg["latent_dim"]))

    def _condition(self, ids, mask, cw, wstart, fw, fp, spk):
        h, dur = self.net.text_stage(ids, mask)
        c = dur.clamp(min=1e-4) * mask
        done = c.cumsum(-1)
        word = cw.clamp(min=0)
        total = torch.zeros_like(c).scatter_add_(1, word, c).gather(1, word)
        cp = ((done - (done - c).gather(1, wstart) - 0.5 * c) / total.clamp(min=1e-8)).clamp(0.0, 1.0) * mask
        return self.net.aligner(h, cw, cp, fw, fp, mask, cw.shape[1]) + self.spk(spk)[:, None]

    def _velocity(self, x, cond, t, fmask):
        cos, sin = rope(torch.arange(x.shape[1], device=x.device), self.net.dh)
        return self.net.backbone(x, cond, t, fmask, cos, sin)

    @torch.inference_mode()
    def generate(self, ids, steps=8, guidance=4.0, speed=1.0, seed=None, speaker=0):
        """ids: letter ids [L] -> latents [T, 64] (midpoint ODE + classifier-free guidance)."""
        dev = ids.device
        text_ids = ids[None]
        cw, ws = word_layout_from_ids(text_ids, self.net.stoi)
        mask = torch.ones_like(text_ids, dtype=torch.bool)
        _, dur = self.net.text_stage(text_ids, mask)
        dur = dur / speed
        counts = torch.zeros(1, int(cw.max()) + 1, device=dev).scatter_add_(1, cw, dur).round().clamp(1, 250).long()[0]
        fw, fp = frame_layout(counts.cpu())
        fw, fp = fw[None].to(dev), fp[None].to(dev)
        fmask = torch.ones_like(fw, dtype=torch.bool)
        cond = self._condition(text_ids, mask, cw, ws, fw, fp, torch.tensor([speaker], device=dev))
        g = torch.Generator(device=dev)
        g.manual_seed(seed if seed is not None else torch.seed() % (2 ** 31))
        x = torch.randn(1, fw.shape[1], self.cfg["latent_dim"], device=dev, generator=g)

        def v(x, t):
            tt = torch.full((1,), t, device=dev)
            vc = self._velocity(x, cond, tt, fmask)
            if guidance == 1.0:
                return vc
            vu = self._velocity(x, torch.zeros_like(cond), tt, fmask)
            return vu + guidance * (vc - vu)

        ts = torch.linspace(0, 1, steps + 1).tolist()
        for a, b in zip(ts[:-1], ts[1:]):
            x = x + (b - a) * v(x + (b - a) / 2 * v(x, a), (a + b) / 2)
        return (x[0] * self.lat_std + self.lat_mean).float()


def word_layout_from_ids(ids, stoi):
    """Word layout from letter ids (the space id marks word ends)."""
    space = stoi[" "]
    row = ids[0].tolist()
    text = "".join(" " if i == space else "x" for i in row)
    cw, ws = word_layout(text)
    return torch.tensor([cw], device=ids.device), torch.tensor([ws], device=ids.device)
