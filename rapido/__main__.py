"""python -m rapido "Olá, tudo bem?" -o ola.wav"""
import argparse

from . import Config, Rapido


def main():
    ap = argparse.ArgumentParser(prog="rapido", description="Brazilian Portuguese text-to-speech")
    ap.add_argument("text", nargs="?", help="text to speak (or use -f)")
    ap.add_argument("-f", "--file", help="read text from a file")
    ap.add_argument("-o", "--out", default="rapido.wav")
    ap.add_argument("--model", default=Config.model)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    text = open(a.file, encoding="utf-8").read() if a.file else a.text
    if not text:
        ap.error("give text or -f FILE")
    tts = Rapido(Config(model=a.model, device=a.device, steps=a.steps, guidance=a.guidance, speed=a.speed, seed=a.seed))
    r = tts.synthesize(text)
    r.save(a.out)
    m = r.metrics
    print(f"{a.out}: {m['audio_s']}s of audio in {m['total_s']}s ({m['x_realtime']}x realtime) on {tts.device}")


if __name__ == "__main__":
    main()
