<p align="center"><img src="assets/logo_banner.png" alt="Rápido TTS" width="100%"></p>

<p align="center">
  <a href="https://huggingface.co/edwixx/rapido-tts"><img src="https://img.shields.io/badge/🤗%20Model-rapido--tts-ffd600" alt="Hugging Face"></a>
  <img src="https://img.shields.io/badge/language-pt--BR-009c3b" alt="pt-BR">
  <img src="https://img.shields.io/badge/params-21M-0b3d2e" alt="21M params">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="Apache 2.0">
</p>

# Rápido TTS

**Tiny, fast Brazilian Portuguese text-to-speech.** 21M parameters, a 42 MB checkpoint, 48 kHz audio and Apache 2.0. It runs offline on your own GPU or CPU, so no text or audio leaves the machine.

- **Reads Brazil right:** numbers, money, dates, times, ordinals, units and phone numbers come out the way a Brazilian says them, with **2.94% WER on our numbers & pronunciation test**. That is the lowest of the small open pt-BR TTS models we tested.
- **Small:** 21M parameters and a 42 MB checkpoint, trained from scratch for Brazilian Portuguese.
- **Fast:** **76× real time** on one GPU, with first audio in **129 ms** and only **1.2 GB** of GPU memory.
- **Private:** it runs fully offline on your GPU or CPU.

## Roadmap

Rápido v1 is the first of a family of small, fast speech models.

- [x] **Rápido v1:** Brazilian Portuguese TTS, 21M parameters, one built-in voice
- [ ] **Rápido Clone:** zero-shot voice cloning from a few seconds of reference audio
- [ ] **Rápido Multilingual:** one compact model for many languages

## Install

```bash
git clone https://github.com/anurag12-webster/rapido-tts && cd rapido-tts
pip install -r requirements.txt
```

Requires Python 3.10+. For a smaller CPU-only install, install PyTorch first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`, then `pip install -r requirements.txt`.

## Use

```python
from rapido import Rapido

tts = Rapido.from_pretrained()                 # downloads the weights once; GPU if available

result = tts.synthesize("Olá! Tudo bem com você?")
result.save("ola.wav")
```

`synthesize()` returns a `Result`:

- `result.output["audio"]` is float32 mono PCM.
- `result.output["sample_rate"]` is 48000.
- `result.output["text"]` is the spoken form the model read.
- `result.metrics` holds the audio length, the time taken and the speed relative to real time.

### Streaming

```python
for chunk in tts.synthesize(long_text, stream=True):
    play(chunk.output["audio"], chunk.output["sample_rate"])
```

Each chunk is one sentence, returned as soon as it is ready. Joined together, the chunks are the full text.

### Settings

```python
from rapido import Rapido, Config

tts = Rapido(Config(device="cpu", steps=8, guidance=4.0, speed=1.0, seed=0, cpu_threads=8))
tts.synthesize("Bom dia!", speed=1.1, seed=7)   # per-call overrides
```

| Setting | Default | |
|---|---|---|
| `steps` | 8 | sampling steps; 8 scores the same as 16 |
| `guidance` | 4.0 | classifier-free guidance; 3 to 4 works best |
| `speed` | 1.0 | speaking rate |
| `seed` | 0 | the same seed gives the same audio; `None` means random |
| `device` | `"auto"` | `"cuda"`, `"cpu"` or `"auto"` |

### Command line

```bash
python -m rapido "O boleto vence no dia 05/11/2026." -o boleto.wav
python -m rapido -f texto.txt -o texto.wav
```

## Portuguese, as written

Write normally. Rápido expands the written form into words before speaking, and `tts.normalize(text)` shows you exactly what it will say.

| Written | Spoken |
|---|---|
| `R$ 1.299,90` | mil duzentos e noventa e nove reais e noventa centavos |
| `05/11/2026` | cinco de novembro de dois mil e vinte e seis |
| `às 14h30` | às catorze horas e trinta minutos |
| `4,5%` | quatro vírgula cinco por cento |
| `200 pessoas` · `2 crianças` | duzentas pessoas · duas crianças |
| `1ª colocada, 3º lugar` | primeira colocada, terceiro lugar |
| `século XVIII` · `Dom Pedro II` | século dezoito · dom pedro segundo |
| `95 km` · `16 °C` | noventa e cinco quilômetros · dezesseis graus celsius |
| `(11) 98765-4321` | um um, nove oito sete seis cinco, quatro três dois um |
| `Sr. Silva, Av. Paulista, nº 1000` | senhor silva, avenida paulista, número mil |

## Benchmarks

Brazilian text is full of things written one way and spoken another: prices, dates, times, percentages, ordinals and phone numbers. We built a 40-sentence test of exactly that, gave every model the same raw text, transcribed the audio with Whisper large-v3, and compared the result word by word (numbers expanded to words before scoring).

| Model | Size | WER on numbers, money & dates ↓ |
|---|---|---|
| **Rápido TTS** | **21M** | **2.94%** |
| Piper pt-BR (faber, medium) | ~16M | 3.31% |
| Kokoro-82M (pf_dora) | 82M | 4.68% |
| MMS-TTS Portuguese | 36M | 14.88% |

## Speed

| | One request | First audio | GPU memory |
|---|---|---|---|
| NVIDIA H100 | **76× real time** | 129 ms | 1.2 GB |
| NVIDIA L4 | **29× real time** | 273 ms | 1.2 GB |
| Laptop CPU (8 threads) | 1.6× real time | 0.6 s | — |

Measured with plain PyTorch at 8 steps, one request at a time, taking the middle of 3 passes after a warm-up.

## How it works

1. A convolutional text encoder reads Portuguese letters.
2. A duration predictor places each letter on a word–letter timeline.
3. A windowed Gaussian aligner gives every audio frame a soft mix of the letters near it.
4. A flow-matching diffusion transformer generates 64-dimensional latents at 25 Hz in 8 steps.
5. The frozen [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) AudioVAE decodes the latents to 48 kHz.

## Limitations

- **One voice.** There is no voice cloning and no emotion control.
- **Portuguese only.** Foreign names, English words and acronyms are read with Portuguese spelling rules.
- **Narration style.** The delivery is calm and read-aloud rather than conversational.
- **Synthetic speech.** Tell listeners when they are hearing an AI voice, and never use it to impersonate anyone.

## Acknowledgements

The 48 kHz decoder is the AudioVAE from [VoxCPM2](https://github.com/OpenBMB/VoxCPM) by OpenBMB (Apache 2.0).

## License

[Apache 2.0](LICENSE).

## Citation

```bibtex
@misc{rapido2026,
  title        = {Rápido: Tiny, Fast Brazilian Portuguese Text-to-Speech},
  author       = {Kanade, Anurag},
  year         = {2026},
  howpublished = {\url{https://github.com/anurag12-webster/rapido-tts}}
}
```
