#!/usr/bin/env python3
"""
Docker Setup Validation Script

Run this to verify that the Docker setup is properly configured.
"""
import os
import sys
from pathlib import Path


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a required file exists."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ Missing {description}: {filepath}")
        return False


def validate_docker_setup():
    """Validate Docker setup configuration."""
    print("🐳 Validating Docker Setup for Federated HVAC System")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    all_good = True
    
    # Check core Docker files
    docker_files = [
        ("Dockerfile", "Docker container configuration"),
        ("docker-compose.yml", "Docker Compose orchestration"),
        (".dockerignore", "Docker ignore file"),
    ]
    
    for filename, description in docker_files:
        if not check_file_exists(project_root / filename, description):
            all_good = False
    
    # Check runner scripts
    print(f"\n📦 Runner Scripts:")
    runner_files = [
        ("src/runners/__init__.py", "Runners module init"),
        ("src/runners/run_coordinator.py", "Coordinator service runner"),
        ("src/runners/run_infrastructure.py", "Infrastructure services runner"),
        ("src/runners/run_sensors.py", "Sensors service runner"),
        ("src/runners/demo_orchestrator.py", "Demo orchestrator"),
    ]
    
    for filename, description in runner_files:
        if not check_file_exists(project_root / filename, description):
            all_good = False
    
    # Check directories
    print(f"\n📁 Required Directories:")
    directories = [
        ("config", "Configuration directory"),
        ("logs", "Logs output directory"),  
        ("charts", "Charts output directory"),
        ("src/actors", "Actors source directory"),
        ("src/federation", "Federation source directory"),
    ]
    
    for dirname, description in directories:
        dirpath = project_root / dirname
        if dirpath.exists() and dirpath.is_dir():
            print(f"✅ {description}: {dirname}/")
        else:
            print(f"❌ Missing {description}: {dirname}/")
            all_good = False
    
    # Check key source files
    print(f"\n🔧 Core Source Files:")
    source_files = [
        ("src/actors/__init__.py", "Base Actor class"),
        ("src/actors/coordinator_actor.py", "Coordinator Actor"),
        ("src/actors/sensor_actor.py", "Sensor Actor"),
        ("src/actors/logger_actor.py", "Logger Actor"),
        ("src/actors/device_controller_actor.py", "Device Controller Actor"),
        ("config/system_config.json", "System configuration"),
        ("requirements.txt", "Python dependencies"),
    ]
    
    for filename, description in source_files:
        if not check_file_exists(project_root / filename, description):
            all_good = False
    
    # Summary
    print(f"\n📊 Validation Summary:")
    if all_good:
        print("✅ All Docker setup files are present!")
        print("\n🚀 Ready to run:")
        print("   docker-compose up --build")
    else:
        print("❌ Some required files are missing.")
        print("   Please check the missing files above.")
    
    return all_good


def show_usage_instructions():
    """Show usage instructions for Docker setup."""
    print(f"\n📖 Usage Instructions:")
    print("=" * 30)
    print("1. Start the system:")
    print("   docker-compose up --build")
    print("")
    print("2. Monitor progress:")
    print("   docker-compose logs -f")
    print("   docker-compose logs -f demo")
    print("")
    print("3. Check results:")
    print("   ls ./logs/")
    print("   python visualization.py")
    print("")
    print("4. Stop the system:")
    print("   docker-compose down")


if __name__ == "__main__":
    success = validate_docker_setup()
    
    if success:
        show_usage_instructions()
        sys.exit(0)
    else:
        sys.exit(1)