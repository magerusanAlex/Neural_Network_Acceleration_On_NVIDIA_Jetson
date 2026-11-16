# S3 - Validarea exportului ONNX

## Scop
Verificam consistenta detectiilor PyTorch si ONNX pe acelasi esantion FLIR RGB.
Acesta este un test preliminar al exportului, nu un benchmark de viteza sau o evaluare mAP.

## Rulare
Din radacina repository-ului, cu mediul proiectului:

```powershell
.\.venv\Scripts\python.exe scripts/validate_onnx.py
```

Scriptul selecteaza implicit 5 imagini distribuite uniform in lista sortata a
fisierelor din images_rgb_val/data. Optiunea `--count 20` extinde esantionul.
Raportul este scris in results/onnx_validation/report.json (exclus din Git).
`--output` permite alegerea altei destinatii.

## Metoda
- Ambele modele ruleaza pe CPU, la 640x640, cu rect=False pentru letterbox patrat.
- Aceleasi praguri: confidence=0.25, NMS IoU=0.7, max_det=300.
- Verificam graful ONNX si egalitatea maparilor claselor.
- Asociem chenarele aceleiasi clase, greedy, in ordinea descrescatoare a scorului
  PyTorch, alegand cel mai mare IoU disponibil (minimum 0.5).
- Testul cere zero detectii neasociate, IoU >= 0.99, diferenta maxima a unei
  coordonate <= 1 pixel si diferenta absoluta de scor <= 0.001.
- Pragurile sunt criterii practice ale acestui test, nu garantii universale ONNX.
- Doua rezultate goale sunt consistente, dar un esantion complet gol nu trece.
- Raportul include setarile, versiunile si hashurile modelelor si imaginilor.

## Limite
Nu comparam cu adnotarile ground truth. Nu demonstram acuratetea modelului,
accelerarea sau compatibilitatea TensorRT pe Jetson. Asocierea greedy poate
necesita analiza manuala in scene aglomerate; o abatere semnalata nu dovedeste
singura ca exportul este gresit. Extinderea esantionului ramane utila.

## Rezultat verificat - 16 noiembrie 2026
Rularea implicita a trecut pe toate cele 5 imagini. Numarul detectiilor a fost
7, 11, 1, 4, 0 in ambele variante: 23 de perechi si zero detectii neasociate.
Raportul JSON local pastreaza diferentele numerice si mediul exact al rularii.
Acest rezultat inchide doar verificarea preliminara pe esantionul ales.
