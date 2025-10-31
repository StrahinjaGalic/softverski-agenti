# Federativno Učenje za Kontrolu HVAC Sistema

Projekat implementira federativno učenje sa aktorskim sistemom za kontrolu zajedničkog uređaja (klima/grejanje) na osnovu senzorskih podataka iz više lokacija.

## Struktura Projekta

```
src/
├── actors/              # Aktorski sistem
│   ├── sensor_actor.py     # SensorActor - lokalno treniranje i senziranje
│   ├── coordinator_actor.py # CoordinatorActor - agregacija modela
│   ├── device_controller_actor.py # DeviceControllerActor - kontrola uređaja  
│   └── logger_actor.py     # LoggerActor - logovanje metrika
├── federation/          # Federativno učenje
│   ├── fedavg.py          # FedAvg algoritam
│   └── model.py           # Linearni regresioni model
├── utils/              # Pomoćne klase
│   ├── messages.py        # Definicije poruka između aktora
│   └── config.py          # Konfiguracija sistema
└── simulation/         # Simulacija i demo
    ├── data_generator.py  # Generisanje test podataka
    └── demo.py           # Glavna demonstracija
```

## Instalacija

```bash
pip install -r requirements.txt
```

## Pokretanje

```bash
python src/simulation/demo.py
```

## Funkcionalnosti

- **5 SensorActor procesa** - mešaju temperaturu i osvetljenost
- **Federativno učenje** - FedAvg algoritam za globalni model
- **Stabilna kontrola** - histereza, min-on/off, ekskluzivni modovi
- **Real-time monitoring** - logovanje i vizuelizacija

## Tehnologije

- Python 3.12
- asyncio za aktorski sistem
- scikit-learn za machine learning
- matplotlib za vizuelizaciju