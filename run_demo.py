#!/usr/bin/env python3
"""
Manual Demo Runner for Dockerized Federated HVAC System
=======================================================

This script allows you to manually run the federated learning demo
when the Docker containers are running.

Usage:
    python run_demo.py

Prerequisites:
    - Docker containers must be running: docker-compose up -d
    - All services (coordinator, sensors, infrastructure) must be healthy

The script will:
    1. Show detailed federation learning progress (rounds, MSE values)
    2. Monitor real-time HVAC control with sensor data
    3. Generate comprehensive log files and results
"""
import subprocess
import sys


def check_containers_running():
    """Check if all required containers are running."""
    try:
        result = subprocess.run(['docker-compose', 'ps', '--services', '--filter', 'status=running'], 
                              capture_output=True, text=True, cwd='.')
        running_services = result.stdout.strip().split('\n')
        
        required_services = ['coordinator', 'sensors', 'infrastructure', 'demo']
        missing_services = [svc for svc in required_services if svc not in running_services]
        
        if missing_services:
            print("❌ Required containers are not running:")
            for svc in missing_services:
                print(f"   - {svc}")
            print("\n💡 Start containers with: docker-compose up -d")
            return False
        
        print("✅ All required containers are running")
        return True
        
    except FileNotFoundError:
        print("❌ docker-compose not found. Make sure Docker is installed.")
        return False
    except Exception as e:
        print(f"❌ Error checking containers: {e}")
        return False


def run_demo():
    """Execute the demo orchestrator in the demo container."""
    print("🚀 Starting Federated HVAC Demo...")
    print("   (Press Ctrl+C to stop)")
    print()
    
    try:
        # Execute the demo orchestrator in the running demo container
        subprocess.run([
            'docker', 'exec', '-it', 
            'softverski-agenti-demo-1', 
            'python', '-m', 'src.runners.demo_orchestrator'
        ], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Demo execution failed with exit code {e.returncode}")
        print("💡 Check if containers are running: docker-compose ps")
        return False
    except KeyboardInterrupt:
        print("\n\n👋 Demo stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        return False
    
    return True


def main():
    """Main execution function."""
    print("🐳 FEDERATED HVAC SYSTEM - MANUAL DEMO RUNNER")
    print("=" * 50)
    
    # Check if containers are running
    if not check_containers_running():
        sys.exit(1)
    
    print()
    
    # Run the demo
    success = run_demo()
    
    if success:
        print("\n🎉 Demo completed!")
        print("📁 Check the 'logs/' directory for detailed results")
        print("📊 system_log.json contains all metrics and events")
    else:
        print("\n💔 Demo failed or was interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()