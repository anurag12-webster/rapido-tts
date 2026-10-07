"""Rápido inference: Brazilian Portuguese text in, 48 kHz speech out."""
import json
import re
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from ._vae import AudioVAE, AudioVAEConfig
from .frontend import VOCAB, encode, normalize
from .model import RapidoModel

DEFAULT_REPO = "edwixx/rapido-tts"
DECODER_REPO = "openbmb/VoxCPM2"  # frozen AudioVAE decoder, Apache 2.0
SAMPLE_RATE = 48000


@dataclass
class Config:
    model: str = DEFAULT_REPO        # HF repo id or a local .pt path
    device: str = "auto"             # "auto" | "cuda" | "cpu"
    steps: int = 8                   # ODE steps (8 matches 16 in accuracy)
    guidance: float = 4.0            # classifier-free guidance
    speed: float = 1.0               # speaking rate
    seed: int | None = 0             # None = random
    cpu_threads: int | None = None


@dataclass
class Result:
    output: dict                     # {"audio": np.float32 [N], "sample_rate": 48000, "text": spoken form}
    metrics: dict = field(default_factory=dict)

    def save(self, path):
        import wave
        pcm = (np.clip(self.output["audio"], -1, 1) * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.output["sample_rate"])
            w.writeframes(pcm.tobytes())
        return path


class Rapido:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        c = self.config
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if c.device == "auto" else c.device
        if c.cpu_threads:
            torch.set_num_threads(c.cpu_threads)
        path = c.model if c.model.endswith(".pt") else hf_hub_download(c.model, "rapido.pt")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model = RapidoModel(ck["cfg"], VOCAB, len(ck["spk_order"]))
        self.model.load_state_dict({k: v.float() for k, v in ck["ema"].items()})
        self.model.to(self.device).eval()
        vae_cfg = json.load(open(hf_hub_download(DECODER_REPO, "config.json")))["audio_vae_config"]
        self.vae = AudioVAE(AudioVAEConfig(**vae_cfg))
        sd = torch.load(hf_hub_download(DECODER_REPO, "audiovae.pth"), map_location="cpu", weights_only=True)
        self.vae.load_state_dict(sd.get("state_dict", sd))
        self.vae.to(self.device).eval()

    @classmethod
    def from_pretrained(cls, model: str = DEFAULT_REPO, **kwargs):
        return cls(Config(model=model, **kwargs))

    def normalize(self, text: str) -> str:
        """The exact spoken form the model reads (numbers, money, dates... expanded in pt-BR)."""
        return normalize(text)

    def synthesize(self, text: str, stream: bool = False, **settings):
        """Text -> Result. With stream=True, yields one Result per sentence as soon as it is ready.
        settings override Config: steps, guidance, speed, seed."""
        gen = self._pieces(text, settings, max_chars=0 if stream else 220)
        return gen if stream else self._join(list(gen))

    @torch.inference_mode()
    def _pieces(self, text, settings, max_chars=220):
        c = self.config
        steps, guidance = settings.get("steps", c.steps), settings.get("guidance", c.guidance)
        speed, seed = settings.get("speed", c.speed), settings.get("seed", c.seed)
        for k, piece in enumerate(_sentences(text, max_chars)):
            spoken = normalize(piece)
            if not spoken:
                continue
            t0 = time.time()
            ids = torch.tensor(encode(spoken), device=self.device)
            z = self.model.generate(ids, steps=steps, guidance=guidance, speed=speed,
                                    seed=None if seed is None else seed + k)
            audio = self.vae.decode(z.T[None]).float().reshape(-1).cpu().numpy()
            dt = time.time() - t0
            dur = len(audio) / SAMPLE_RATE
            yield Result({"audio": audio, "sample_rate": SAMPLE_RATE, "text": spoken},
                         {"audio_s": round(dur, 3), "total_s": round(dt, 3), "x_realtime": round(dur / dt, 2)})

    @staticmethod
    def _join(parts, pause=0.22):
        if not parts:
            raise ValueError("nothing to say after normalization")
        gap = np.zeros(int(pause * SAMPLE_RATE), np.float32)
        audio = np.concatenate([x for p in parts for x in (p.output["audio"], gap)][:-1])
        total = sum(p.metrics["total_s"] for p in parts)
        dur = len(audio) / SAMPLE_RATE
        return Result({"audio": audio, "sample_rate": SAMPLE_RATE, "text": " ".join(p.output["text"] for p in parts)},
                      {"audio_s": round(dur, 3), "total_s": round(total, 3), "x_realtime": round(dur / total, 2)})


def _sentences(text, max_chars=220):
    out, cur = [], ""
    for p in re.split(r"(?<=[.!?…])\s+", text.strip()):
        if cur and len(cur) + len(p) > max_chars:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    return [p for p in out + [cur] if p.strip()]
