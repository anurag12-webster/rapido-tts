import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from anywhere

from rapido import Rapido

tts = Rapido.from_pretrained()  # downloads the model once; uses the GPU if there is one

result = tts.synthesize("Olá! O boleto de R$ 1.299,90 vence no dia 05/11/2026, às 14h30.")
result.save("ola.wav")
print(result.output["text"])  # the spoken form the model read
print(result.metrics)          # audio length, time taken, x realtime

# sentence-by-sentence, for live playback
for chunk in tts.synthesize("Bom dia! Tudo bem com você? Já são quase oito horas.", stream=True):
    print(chunk.output["text"], chunk.metrics)
