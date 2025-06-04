# Cable Analysis CNN

Šiame projekte pateikiamas paprastas 1D CNN modelis, skirtas elektros kabelio bandymo duomenims klasifikuoti. Modelis priima PD, FDS, PDC ir RMV matavimų sekas ir pateikia išvadą apie kabelio būklę:

1. **Geras** – kabelis tinkamas naudoti.
2. **Reikia papildomo bandymo** – rekomenduojama pakartoti bandymą po 6 mėnesių.
3. **Blogas** – būtina lokalizuoti gedimo tašką ir remontuoti.

## Duomenų paruošimas

Mokymo duomenys turi būti pateikti CSV faile su šiomis stulpeliais:

```
PD,FDS,PDC,RMV,label
```

- `PD`, `FDS`, `PDC`, `RMV` – matavimų sekų reikšmės (vienas įrašas sudaro `seq_length` imčių).
- `label` – klasė: `0` geras, `1` reikia papildomo bandymo, `2` blogas.

Duomenys CSV faile turi būti išdėstyti taip, kad kiekvienas mėginys būtų `seq_length` iš eilės einančių eilučių.

## Mokymas

```
python -m cable_analysis.train data.csv --epochs 20 --batch-size 64 --output model.pt
```

## Klasifikavimas

```
python -m cable_analysis.predict model.pt sample.npy
```

`sample.npy` turi būti numpy masyvas su forma `(4, seq_length)`.
