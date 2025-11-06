"""
Runners module - Service entry points for Docker containers.

Each runner is responsible for starting specific actors in their designated containers:
- run_infrastructure.py: LoggerActor + DeviceControllerActor
- run_coordinator.py: CoordinatorActor  
- run_sensors.py: All 5 SensorActors
- demo_orchestrator.py: Demo orchestration logic (replaces demo.py)
"""