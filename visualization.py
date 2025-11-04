"""
VISUALIZATION DASHBOARD - Grafički prikaz rezultata

Učitava logove iz `logs/system_log.json` i kreira vizualizacije:
1. MSE konvergencija tokom federacije
2. Timeline temperatura senzora
3. Timeline HVAC moda (IDLE/COOL/HEAT)
4. Komande poslate DeviceController-u
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, 'd:/4.godina/Agenti/Projekat/softverski-agenti/src')


class VisualizationDashboard:
    """Dashboard za vizualizaciju rezultata sistema."""
    
    def __init__(self, log_file: str = "logs/system_log.json"):
        self.log_file = Path(log_file)
        self.data = None
        
    def load_data(self):
        """Učitaj logove."""
        if not self.log_file.exists():
            print(f"❌ Log fajl ne postoji: {self.log_file}")
            return False
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Flatten metrics dictionary to list
        metrics = self.data.get('metrics', {})
        if isinstance(metrics, dict):
            self.metrics_list = []
            for metric_type, metric_array in metrics.items():
                self.metrics_list.extend(metric_array)
        else:
            self.metrics_list = metrics
        
        print(f"✓ Učitano iz: {self.log_file}")
        print(f"  - Metrika: {len(self.metrics_list)}")
        print(f"  - Događaja: {len(self.data.get('events', []))}\n")
        return True
    
    def plot_federation_convergence(self):
        """Graf 1: MSE konvergencija tokom federacije."""
        metrics = self.metrics_list
        
        # Izvuci aggregation metrike
        aggregations = [m for m in metrics if m.get('type') == 'aggregation']
        
        if not aggregations:
            print("⚠️  Nema aggregation metrika za prikaz")
            return
        
        rounds = []
        mse_values = []
        participants = []
        
        for agg in aggregations:
            round_num = agg.get('round', 0)
            data = agg.get('data', {})
            
            rounds.append(round_num)
            
            # Koristi avg_mse ako postoji, inače računaj weight magnitude
            if 'avg_mse' in data:
                mse_values.append(data['avg_mse'])
            else:
                weights = np.array(data.get('weights', [0, 0]))
                weight_magnitude = np.linalg.norm(weights)
                mse_values.append(weight_magnitude)
            
            participants.append(data.get('num_participants', 0))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # MSE konvergencija
        ylabel = 'Average MSE' if aggregations[0].get('data', {}).get('avg_mse') is not None else 'Weight Magnitude'
        ax1.plot(rounds, mse_values, marker='o', linewidth=2, markersize=8, color='#2E86AB')
        ax1.set_xlabel('Federation Round', fontsize=12)
        ax1.set_ylabel(ylabel, fontsize=12)
        ax1.set_title('Model Convergence During Federation', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(rounds)
        
        # Annotate final value
        if mse_values:
            final_val = mse_values[-1]
            ax1.annotate(f'Final: {final_val:.4f}', 
                        xy=(rounds[-1], mse_values[-1]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Broj participanata
        ax2.bar(rounds, participants, color='#A23B72', alpha=0.7, width=0.5)
        ax2.set_xlabel('Federation Round', fontsize=12)
        ax2.set_ylabel('Number of Participants', fontsize=12)
        ax2.set_title('Sensor Participation', fontsize=14, fontweight='bold')
        ax2.set_xticks(rounds)
        ax2.set_ylim([0, max(participants) + 1])
        
        plt.tight_layout()
        output_file = 'visualization_federation.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Graf sačuvan: {output_file}")
        plt.show()
    
    def plot_device_commands(self):
        """Graf 2: HVAC komande timeline."""
        metrics = self.metrics_list
        
        # Izvuci device_command metrike
        device_commands = [m for m in metrics if m.get('type') == 'device_command']
        
        if not device_commands:
            print("⚠️  Nema device_command metrika za prikaz")
            return
        
        timestamps = []
        modes = []
        setpoints = []
        
        for cmd in device_commands:
            ts_str = cmd.get('timestamp', '')
            ts = datetime.fromisoformat(ts_str)
            timestamps.append(ts)
            
            data = cmd.get('data', {})
            new_mode = data.get('new_mode', 'IDLE')
            setpoint = data.get('setpoint', 22.0)
            
            modes.append(new_mode)
            setpoints.append(setpoint)
        
        if not timestamps:
            print("⚠️  Nema validnih timestamps")
            return
        
        # Relativno vreme (sekunde od prvog event-a)
        start_time = timestamps[0]
        relative_times = [(t - start_time).total_seconds() for t in timestamps]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Mode timeline
        mode_map = {'IDLE': 0, 'COOL': 1, 'HEAT': 2}
        mode_values = [mode_map.get(m, 0) for m in modes]
        
        colors = {'IDLE': '#A8DADC', 'COOL': '#457B9D', 'HEAT': '#E63946'}
        mode_colors = [colors.get(m, 'gray') for m in modes]
        
        ax1.scatter(relative_times, mode_values, c=mode_colors, s=200, marker='s', edgecolors='black', linewidths=1.5)
        ax1.set_ylabel('HVAC Mode', fontsize=12)
        ax1.set_yticks([0, 1, 2])
        ax1.set_yticklabels(['IDLE', 'COOL', 'HEAT'])
        ax1.set_title('HVAC Mode Timeline', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Setpoint timeline
        ax2.plot(relative_times, setpoints, marker='o', linewidth=2, markersize=8, color='#F77F00')
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Setpoint (°C)', fontsize=12)
        ax2.set_title('Setpoint Timeline', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Annotate
        for i, (t, sp) in enumerate(zip(relative_times, setpoints)):
            ax2.annotate(f'{sp:.1f}°C', xy=(t, sp), xytext=(0, 10),
                        textcoords='offset points', ha='center', fontsize=9)
        
        plt.tight_layout()
        output_file = 'visualization_device.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Graf sačuvan: {output_file}")
        plt.show()
    
    def plot_summary(self):
        """Graf 3: Sumarna statistika."""
        metrics = self.metrics_list
        events = self.data.get('events', [])
        
        # Broji tipove
        from collections import Counter
        metric_types = Counter([m.get('type', 'unknown') for m in metrics])
        event_types = Counter([e.get('event_type', 'unknown') for e in events])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Metrike po tipu
        if metric_types:
            types = list(metric_types.keys())
            counts = list(metric_types.values())
            
            ax1.barh(types, counts, color='#06FFA5', edgecolor='black', linewidth=1.5)
            ax1.set_xlabel('Count', fontsize=12)
            ax1.set_title('Metrics by Type', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Annotate counts
            for i, (t, c) in enumerate(zip(types, counts)):
                ax1.text(c + 0.1, i, str(c), va='center', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No metrics', ha='center', va='center', transform=ax1.transAxes)
        
        # Eventi po tipu
        if event_types:
            types = list(event_types.keys())
            counts = list(event_types.values())
            
            ax2.barh(types, counts, color='#FFBA08', edgecolor='black', linewidth=1.5)
            ax2.set_xlabel('Count', fontsize=12)
            ax2.set_title('Events by Type', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Annotate counts
            for i, (t, c) in enumerate(zip(types, counts)):
                ax2.text(c + 0.1, i, str(c), va='center', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No events', ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        output_file = 'visualization_summary.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Graf sačuvan: {output_file}")
        plt.show()
    
    def generate_all(self):
        """Generiši sve vizualizacije."""
        print("\n" + "="*70)
        print("VISUALIZATION DASHBOARD".center(70))
        print("="*70 + "\n")
        
        if not self.load_data():
            return
        
        print("📊 Generisanje grafova...\n")
        
        try:
            self.plot_federation_convergence()
            self.plot_device_commands()
            self.plot_summary()
            
            print("\n" + "="*70)
            print("✓ Svi grafovi uspešno generisani!".center(70))
            print("="*70 + "\n")
            
        except Exception as e:
            print(f"\n❌ Greška prilikom generisanja grafova: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Entry point."""
    import sys
    
    log_file = "logs/system_log.json"
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    
    dashboard = VisualizationDashboard(log_file)
    dashboard.generate_all()


if __name__ == "__main__":
    main()
