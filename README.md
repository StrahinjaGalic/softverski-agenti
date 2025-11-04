# Federativno Učenje za Kontrolu HVAC Sistema

Projekat implementira federativno učenje sa aktorskim sistemom za kontrolu zajedničkog uređaja (klima/grejanje) na osnovu senzorskih podataka iz više lokacija.

## 🎯 Ključne Features

- ✅ **Federativno učenje** - FedAvg algoritam, 5 distribuiranih senzora
- ✅ **Real-time agregacija** - Median temperatura/predloga, HVAC kontrola
- ✅ **Hysteresis logika** - Deadband, min-on/off times, exclusive control
- ✅ **Actor-based system** - Async TCP komunikacija, robusni protokol
- ✅ **Kompletno testiran** - Integration tests, unit tests
- ✅ **Vizualizacija** - MSE convergence, timeline grafovi

## 📁 Struktura Projekta

```
src/
├── actors/              # Aktorski sistem
│   ├── sensor_actor.py     # SensorActor - lokalno treniranje i senziranje
│   ├── coordinator_actor.py # CoordinatorActor - FedAvg i real-time agregacija
│   ├── device_controller_actor.py # DeviceControllerActor - HVAC kontrola
│   └── logger_actor.py     # LoggerActor - logovanje metrika i događaja
├── federation/          # Federativno učenje
│   ├── fedavg.py          # FedAvg algoritam implementacija
│   └── model.py           # Linearni regresioni model
├── simulation/          # Generisanje podataka
│   └── data_generator.py  # Simulacija senzorskih podataka
└── utils/              # Pomoćne klase
    ├── messages.py        # Message contracts (ModelUpdate, SensorData, ApplyCommand...)
    └── config.py          # System i sensor konfiguracija

tests/
├── test_integration.py    # Kompletan sistem test (federation + real-time)
└── test_device_controller.py # DeviceController unit test

demo.py                 # 🎬 DEMO orchestration script
visualization.py        # 📊 Dashboard za vizualizaciju rezultata
```

## 🚀 Instalacija

```bash
# 1. Kloniraj repo
git clone https://github.com/StrahinjaGalic/softverski-agenti.git
cd softverski-agenti

# 2. Instaliraj dependencies
pip install -r requirements.txt
```

## ▶️ Pokretanje

### 1. **Demo prezentacija** (preporučeno)

```bash
python demo.py
```

Demo prikazuje:
- ⚙️ Pokretanje svih aktera (Coordinator, Logger, DeviceController, 5 senzora)
- 📚 Federativno učenje (3 runde, FedAvg agregacija)
- 🔄 Real-time agregacija (5 ciklusa, HVAC komande)
- 📊 Finalni rezultati (MSE, komande, logovi)

### 2. **Integration testovi**

```bash
# Windows PowerShell
$env:PYTHONPATH='src'; $env:PYTHONIOENCODING='utf-8'; python tests/test_integration.py

# Linux/Mac
PYTHONPATH=src python tests/test_integration.py
```

### 3. **Vizualizacija rezultata**

```bash
python visualization.py
```

Generiše 3 grafa:
- 📈 **MSE convergence** tokom federacije
- 🎛️ **HVAC mode timeline** (IDLE/COOL/HEAT)
- 📊 **Summary statistika** (metrike i događaji)

## 🏗️ Arhitektura Sistema

### **Akteri:**

1. **CoordinatorActor** (port 8000)
   - Hub za federativno učenje i real-time agregaciju
   - FedAvg: Prikuplja ModelUpdate → agregira težine → broadcast GlobalModelUpdate
   - Real-time: Prima SensorData → prikuplja predloge → šalje ApplyCommand

2. **SensorActor** (ports 8010-8014)
   - Lokalno treniranje linearnog modela (temperatura, luminozitet → setpoint)
   - Simulacija senzorskih podataka (T, L)
   - Real-time predlozi setpoint-a na osnovu trenutnih uslova

3. **DeviceControllerActor** (port 8001)
   - Hysteresis kontrola: deadband (±0.5°C), min-on time (2min), min-off time (1min)
   - Exclusive control: HEAT ↔ COOL ne može direktno (mora kroz IDLE)
   - State machine: IDLE → COOL/HEAT sa validacijom

4. **LoggerActor** (port 8002)
   - Prikuplja metrike (aggregation, mse, device_command)
   - Prikuplja događaje (federation_complete, mode_change)
   - Čuva logove u `logs/system_log.json`

### **Communication Protocol:**

- **Length-prefixed TCP**: 4 bytes (int32 big-endian) + JSON payload
- Robusna komunikacija, error handling
- Message types: StartTraining, ModelUpdate, GlobalModelUpdate, SensorData, ApplyCommand, LogMetrics

## 📊 Faze Izvršavanja

### **FAZA 1: Federativno Učenje**
1. Coordinator šalje `StartTraining` → svi senzori
2. Senzori treniraju lokalne modele
3. Senzori šalju `ModelUpdate` (weights, bias, MSE) → Coordinator
4. Coordinator agregira pomoću FedAvg algoritma
5. Coordinator broadcast `GlobalModelUpdate` → svi senzori
6. Ponavljanje za N rundi

### **FAZA 2: Real-time Agregacija**
1. Senzori šalju `SensorData` (T, L) → Coordinator
2. Coordinator šalje `CollectProposals` → senzori
3. Senzori vraćaju `ProposalResponse` (predlog setpoint-a)
4. Coordinator:
   - Računa T_med = median(temperature)
   - Računa Y_agg = median(proposals)
   - Određuje mode: COOL ako T_med > Y_agg+threshold, HEAT ako <, IDLE otherwise
5. Coordinator šalje `ApplyCommand` (mode, setpoint) → DeviceController
6. DeviceController validira sa hysteresis logikom i primenjuje

## 🧪 Testiranje

```bash
# Integration test (kompletan sistem)
pytest tests/test_integration.py -v

# DeviceController unit test
pytest tests/test_device_controller.py -v

# Svi testovi
pytest tests/ -v
```

**Testovi pokrivaju:**
- ✅ Federation: 3 runde, 5 senzora, MSE konvergencija
- ✅ Real-time: Agregacija, mode determination, command sending
- ✅ Hysteresis: Deadband, min-on/off times, exclusive control
- ✅ Logging: Metrics i events

## 📈 Rezultati

**Federation MSE (tipično):**
- Runda 1: ~0.045
- Runda 2: ~0.040
- Runda 3: ~0.038 (konvergira)

**Real-time:**
- 2/3 ciklusa uspešno (prvi ciklus - senzori se pripremaju)
- HVAC mode: COOL @ 23.5°C (za T_med ~25°C)

## 🛠️ Tehnologije

- **Python 3.11+**
- **asyncio** - Aktorski sistem, async TCP
- **scikit-learn** - LinearRegression model
- **numpy** - Numeričke operacije, FedAvg
- **matplotlib** - Vizualizacija grafova
- **pytest** - Unit i integration testovi

## 📝 Logovi

Logovi se čuvaju u `logs/system_log.json`:

```json
{
  "metrics": [
    {
      "timestamp": "2025-11-04T15:45:00.868",
      "metric_type": "aggregation",
      "value": 0.0383,
      "round_number": 3,
      "data": {"participants": 5, "mse": 0.0383}
    },
    {
      "timestamp": "2025-11-04T15:45:07.474",
      "metric_type": "device_command",
      "value": 1.0,
      "data": {"old_mode": "IDLE", "new_mode": "COOL", "setpoint": 23.5}
    }
  ],
  "events": [
    {
      "timestamp": "2025-11-04T15:45:03.404",
      "event_type": "federation_complete",
      "description": "Completed 3 rounds"
    }
  ]
}
```

## 👥 Autori

- **Strahinja Galic** - Demo script, Visualization dashboard, Integration tests, Documentation
- **Mihajlo Sremac** - CoordinatorActor, DeviceControllerActor, LoggerActor, Communication protocol


## 📄 License

MIT License
