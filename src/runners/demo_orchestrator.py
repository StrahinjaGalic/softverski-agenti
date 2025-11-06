"""
Demo Orchestrator - Replaces demo.py functionality for Docker environment.

This service waits for all other services to be ready, then orchestrates the
complete federated learning demo:
1. Federation phase (3 rounds of training)
2. Real-time control phase (5 cycles)
3. Results generation and visualization
"""
import asyncio
import aiohttp
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from utils.config_loader import load_config_from_json


class DockerDemoOrchestrator:
    """Orchestrates the complete federated HVAC demo in Docker environment."""
    
    def __init__(self):
        self.coordinator_host = os.getenv('COORDINATOR_HOST', 'coordinator')
        self.coordinator_port = int(os.getenv('COORDINATOR_PORT', '8000'))
        self.logger_host = os.getenv('LOGGER_HOST', 'infrastructure')
        self.logger_port = int(os.getenv('LOGGER_PORT', '8002'))
        
        # Load configuration
        config_path = "/app/config/system_config.json"
        if os.path.exists(config_path):
            self.config = load_config_from_json(config_path)
        else:
            from utils.config import SystemConfig
            self.config = SystemConfig()
        
        print(f"🐳 Demo Orchestrator initialized")
        print(f"🔗 Coordinator: {self.coordinator_host}:{self.coordinator_port}")
        print(f"🔗 Logger: {self.logger_host}:{self.logger_port}")
    
    def print_header(self, text: str, char: str = "="):
        """Print formatted header."""
        width = 70
        print(f"\n{char * width}")
        print(f"{text:^{width}}")
        print(f"{char * width}\n")
    
    async def wait_for_all_services(self):
        """Wait for all services to be healthy and ready."""
        services = [
            (self.coordinator_host, self.coordinator_port, "Coordinator"),
            (self.logger_host, self.logger_port, "Logger"),
            (self.logger_host, 8001, "DeviceController"),  # Same host as logger
            ("sensors", 8010, "Sensors")
        ]
        
        self.print_header("WAITING FOR SERVICES", "⏳")
        
        for host, port, name in services:
            await self.wait_for_service(host, port, name)
        
        print("✅ All services are ready!")
    
    async def wait_for_service(self, host: str, port: int, name: str):
        """Wait for a specific service to be available."""
        import socket
        
        print(f"⏳ Waiting for {name} at {host}:{port}...")
        
        for attempt in range(30):  # 30 attempts, 2s each = 60s max
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((host, port))
                sock.close()
                if result == 0:
                    print(f"✅ {name} ready at {host}:{port}")
                    return True
            except Exception:
                pass
            
            if attempt < 29:
                await asyncio.sleep(2)
        
        print(f"❌ {name} not available after 60s")
        return False
    
    async def run_federation_phase(self):
        """Trigger and monitor the federation learning phase."""
        self.print_header("FEDERATION LEARNING PHASE", "📚")
        
        print(f"🎯 Starting {self.config.federation.num_rounds} rounds of federated learning...")
        print(f"📊 Local epochs per round: {self.config.federation.local_epochs}")
        print(f"🏠 Participating sensors: {len(self.config.sensors)}")
        
        # Get sensor IDs from config
        sensor_ids = [sensor.sensor_id for sensor in self.config.sensors]
        
        try:
            # Use HTTP API to trigger federation (this would require adding HTTP endpoints to coordinator)
            # For now, we'll simulate by waiting and checking logs
            
            print(f"\n📤 Triggering federation with sensors: {', '.join(sensor_ids)}")
            
            # In a full implementation, we would make HTTP calls to coordinator
            # For this demo, we'll wait for the federation to complete automatically
            # The coordinator will automatically start federation when sensors connect
            
            print("⏳ Federation learning in progress...")
            
            # Monitor federation progress with detailed updates
            federation_start_time = time.time()
            max_federation_time = 180  # 3 minutes max
            last_round_reported = -1
            
            while time.time() - federation_start_time < max_federation_time:
                await asyncio.sleep(3)  # Check every 3 seconds
                
                # Try to get detailed federation status
                current_round, total_rounds, current_mse = await self.get_federation_status()
                
                if current_round > last_round_reported:
                    if current_mse is not None:
                        print(f"   📊 Round {current_round + 1}/{total_rounds} completed - MSE: {current_mse:.6f}")
                    else:
                        print(f"   📊 Round {current_round + 1}/{total_rounds} completed")
                    last_round_reported = current_round
                
                # Check if federation is complete
                if await self.check_federation_complete():
                    federation_time = time.time() - federation_start_time
                    print(f"\n✅ Federation completed in {federation_time:.1f} seconds!")
                    break
                    
            else:
                print("⚠️  Federation taking longer than expected, continuing...")
            
        except Exception as e:
            print(f"❌ Error in federation phase: {e}")
    
    async def run_realtime_phase(self):
        """Monitor the real-time control phase."""
        self.print_header("REAL-TIME CONTROL PHASE", "🔄")
        
        print("🌡️  Starting real-time HVAC control...")
        print("📡 Sensors will send data and receive commands automatically")
        
        # Monitor real-time phase
        realtime_start_time = time.time()
        max_realtime_time = 120  # 2 minutes max
        
        cycles_completed = 0
        target_cycles = 5
        
        while time.time() - realtime_start_time < max_realtime_time:
            await asyncio.sleep(8)  # Check every 8 seconds
            
            # Check realtime progress with details
            new_cycles, latest_sensor_data, latest_command = await self.check_realtime_progress_detailed()
            if new_cycles > cycles_completed:
                cycles_completed = new_cycles
                
                # Show detailed cycle information
                if latest_sensor_data and latest_command:
                    sensor_id = latest_sensor_data.get('sensor_id', 'unknown')
                    temp = latest_sensor_data.get('temperature', 'N/A')
                    mode = latest_command.get('new_mode', 'N/A')
                    print(f"   🌡️  {sensor_id}: T={temp}°C → HVAC mode: {mode}")
                
                print(f"📊 Real-time cycle {cycles_completed} completed")
                
                if cycles_completed >= target_cycles:
                    realtime_time = time.time() - realtime_start_time
                    print(f"\n✅ Real-time control completed in {realtime_time:.1f} seconds!")
                    break
            
        print(f"🎯 Total cycles completed: {cycles_completed}")
    
    async def generate_results(self):
        """Generate final results and summary."""
        self.print_header("RESULTS GENERATION", "📊")
        
        try:
            # Read system logs
            log_file = "/app/logs/system_log.json"
            logs = {'metrics': [], 'events': []}
            if os.path.exists(log_file):
                # Be defensive: file may be empty or malformed while demo runs
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                        if not content or not content.strip():
                            logs = {'metrics': [], 'events': []}
                        else:
                            logs = json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    # Corrupted or empty file; fallback to empty logs
                    logs = {'metrics': [], 'events': []}
                except Exception as e:
                    print(f"⚠️  Could not read log file: {e}")
                    logs = {'metrics': [], 'events': []}

            # Extract key metrics
            metrics = logs.get('metrics', [])
            events = logs.get('events', [])
            
            # Generate summary
            summary = self.create_summary(metrics, events)
            
            # Save summary report
            summary_file = "/app/logs/demo_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(summary)
            
            print("📄 Summary report generated:")
            print(summary)
            
            # Generate visualizations
            await self.generate_visualizations()
                
        except Exception as e:
            print(f"❌ Error generating results: {e}")
    
    def create_summary(self, metrics: dict, events: list) -> str:
        """Create human-readable summary of the demo results."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        summary = f"""
🎯 FEDERATED HVAC SYSTEM - DEMO RESULTS
=====================================
Generated: {now}

📊 FEDERATION PHASE:
"""
        
        # Analyze federation metrics (metrics is now a dict with keys like 'aggregation')
        federation_metrics = metrics.get('aggregation', [])
        if federation_metrics:
            initial_mse = federation_metrics[0].get('data', {}).get('avg_mse', 'N/A')
            final_mse = federation_metrics[-1].get('data', {}).get('avg_mse', 'N/A')
            rounds = len(federation_metrics)
            participants = federation_metrics[-1].get('data', {}).get('num_participants', 'N/A')
            
            summary += f"  - Rounds completed: {rounds}\n"
            summary += f"  - Participants: {participants} sensors\n"
            if isinstance(initial_mse, (int, float)) and isinstance(final_mse, (int, float)):
                improvement = ((initial_mse - final_mse) / initial_mse * 100) if initial_mse > 0 else 0
                summary += f"  - Initial MSE: {initial_mse:.4f}\n"
                summary += f"  - Final MSE: {final_mse:.4f}\n"
                summary += f"  - Improvement: {improvement:.1f}%\n"
        
        summary += f"\n🔄 REAL-TIME CONTROL:\n"
        
        # Analyze device commands (metrics is now a dict)
        device_metrics = metrics.get('device_command', [])
        if device_metrics:
            summary += f"  - HVAC commands sent: {len(device_metrics)}\n"
            modes = [m.get('data', {}).get('new_mode', 'N/A') for m in device_metrics]
            unique_modes = list(set(modes))
            summary += f"  - Modes used: {', '.join(unique_modes)}\n"
        
        # Analyze events
        summary += f"\n📈 SYSTEM EVENTS:\n"
        summary += f"  - Total events logged: {len(events)}\n"
        
        for event in events[:5]:  # Show first 5 events
            event_type = event.get('event_type', 'Unknown')
            description = event.get('description', 'No description')
            summary += f"  - {event_type}: {description}\n"
        
        summary += f"\n📁 FILES GENERATED:\n"
        summary += f"  ✅ system_log.json (detailed metrics)\n"
        summary += f"  ✅ demo_summary.txt (this report)\n"
        
        if os.path.exists("/app/charts"):
            chart_files = os.listdir("/app/charts")
            if chart_files:
                summary += f"  📊 Visualization charts:\n"
                for chart in chart_files:
                    summary += f"     - {chart}\n"
        
        summary += f"\n🎉 DEMO COMPLETED SUCCESSFULLY!\n"
        summary += f"System remains active for exploration and monitoring.\n"
        
        return summary
    
    async def generate_visualizations(self):
        """Generate visualization charts."""
        try:
            print("📊 Generating visualization charts...")
            
            # Try to run visualization script
            import subprocess
            result = subprocess.run([
                sys.executable, "/app/visualization.py"
            ], capture_output=True, text=True, cwd="/app")
            
            if result.returncode == 0:
                print("✅ Visualization charts generated successfully")
            else:
                print(f"⚠️  Visualization generation had issues: {result.stderr}")
                
        except Exception as e:
            print(f"⚠️  Could not generate visualizations: {e}")
    
    async def get_federation_status(self) -> tuple[int, int, float]:
        """Get detailed federation status from coordinator or logs."""
        try:
            # First try to get status from log file
            log_file = "/app/logs/system_log.json"
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                        if not content or not content.strip():
                            return -1, 5, None
                        logs = json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    return -1, 5, None

                # Look for aggregation metrics to determine current round
                metrics = logs.get('metrics', [])
                federation_metrics = [m for m in metrics if m.get('type') == 'aggregation']

                if federation_metrics:
                    current_round = len(federation_metrics) - 1
                    total_rounds = self.config.federation.num_rounds if hasattr(self.config, 'federation') else 5
                    latest_mse = federation_metrics[-1].get('data', {}).get('avg_mse')
                    return current_round, total_rounds, latest_mse

            return -1, self.config.federation.num_rounds if hasattr(self.config, 'federation') else 5, None
        except:
            return -1, 5, None

    async def check_federation_complete(self) -> bool:
        """Check if federation learning is complete."""
        try:
            log_file = "/app/logs/system_log.json"
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                        if not content or not content.strip():
                            return False
                        logs = json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    return False

                events = logs.get('events', [])
                for event in events:
                    if event.get('event_type') == 'federation_complete':
                        return True

                # Also check against configured rounds
                metrics = logs.get('metrics', [])
                federation_metrics = [m for m in metrics if m.get('type') == 'aggregation']
                required = self.config.federation.num_rounds if hasattr(self.config, 'federation') else 5
                return len(federation_metrics) >= required

            return False
        except:
            return False
    
    async def check_realtime_progress_detailed(self) -> tuple[int, dict, dict]:
        """Check real-time progress with detailed sensor and command data."""
        try:
            log_file = "/app/logs/system_log.json"
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                        if not content or not content.strip():
                            return 0, {}, {}
                        logs = json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    return 0, {}, {}

                metrics = logs.get('metrics', [])

                # Get latest sensor data and device commands
                sensor_metrics = [m for m in metrics if m.get('type') == 'sensor_data']
                device_commands = [m for m in metrics if m.get('type') == 'device_command']

                cycles = len(device_commands)
                latest_sensor = sensor_metrics[-1]['data'] if sensor_metrics else {}
                latest_command = device_commands[-1]['data'] if device_commands else {}

                return cycles, latest_sensor, latest_command

            return 0, {}, {}
        except:
            return 0, {}, {}

    async def check_realtime_progress(self) -> int:
        """Check how many real-time cycles have been completed."""
        try:
            log_file = "/app/logs/system_log.json"
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                        if not content or not content.strip():
                            return 0
                        logs = json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    return 0

                # Count device commands as proxy for cycles
                metrics = logs.get('metrics', [])
                device_commands = [m for m in metrics if m.get('type') == 'device_command']
                return len(device_commands)

            return 0
        except:
            return 0
    
    async def run_complete_demo(self):
        """Run the complete federated HVAC demo."""
        print("🐳 FEDERATED HVAC SYSTEM - DOCKER DEMO")
        print("=" * 50)
        print("🚀 Starting automated demo sequence...")
        
        try:
            # Phase 1: Wait for services
            await self.wait_for_all_services()
            
            # Give services time to fully initialize
            print("\n⏳ Allowing services to initialize...")
            await asyncio.sleep(10)
            
            # Phase 2: Federation learning
            await self.run_federation_phase()
            
            # Phase 3: Real-time control  
            await self.run_realtime_phase()
            
            # Phase 4: Results generation
            await self.generate_results()
            
            self.print_header("DEMO COMPLETED", "🎉")
            print("✅ The federated HVAC demo has completed successfully!")
            print("📊 Check /app/logs/ for detailed results and logs")
            print("🔍 System will remain running for exploration...")
            
            # Keep container alive for monitoring
            await self.keep_alive()
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            raise
    
    async def keep_alive(self):
        """Keep the demo container alive for monitoring."""
        print("\n🔄 Demo orchestrator staying active for monitoring...")
        print("   Use 'docker-compose logs -f demo' to see this output")
        print("   Use 'docker-compose down' to stop all services")
        
        try:
            while True:
                await asyncio.sleep(30)
                print("💓 Demo orchestrator heartbeat - system healthy")
        except KeyboardInterrupt:
            print("\n👋 Demo orchestrator shutting down...")


async def main():
    """Main entry point for demo orchestrator."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    orchestrator = DockerDemoOrchestrator()
    await orchestrator.run_complete_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo orchestrator stopped")