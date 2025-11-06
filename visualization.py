"""
VISUALIZATION DASHBOARD - Grafički prikaz rezultata

Učitava logove iz `logs/system_log.json` i kreira vizualizacije:
1. MSE konvergencija tokom federacije
2. Timeline temperatura senzora
3. Timeline HVAC moda (IDLE/COOL/HEAT)
4. Komande poslate DeviceController-u
"""
# Add src to path dynamically
import sys
from pathlib import Path
sys.path.insert(0, str((Path(__file__).parent / 'src').resolve()))

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd


class VisualizationDashboard:
    """Dashboard za vizualizaciju rezultata sistema."""
    
    def __init__(self, log_file: str = "logs/system_log.json"):
        self.log_file = Path(log_file)
        self.data = None
        self.interactive_mode = True  # Toggle between Matplotlib and Plotly
        
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
    
    def create_interactive_learning_dashboard(self):
        """📊 PLOTLY: Interactive Learning Progress Dashboard."""
        metrics = self.metrics_list
        
        # Extract all data types
        aggregations = [m for m in metrics if m.get('type') == 'aggregation']
        local_training = [m for m in metrics if m.get('type') == 'local_training']
        sensor_data = [m for m in metrics if m.get('type') == 'sensor_data']
        device_commands = [m for m in metrics if m.get('type') == 'device_command']
        
        if not aggregations:
            print("⚠️  No aggregation data for learning dashboard")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('🎯 Federation Learning Curve', '🏠 Sensor Participation', 
                          '📈 Local Training Progress', '🌡️ Temperature Proposals'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        # 1. Federation Learning Curve (MSE improvement)
        rounds = [agg.get('round', 0) for agg in aggregations]
        mse_values = []
        participants = []
        
        for agg in aggregations:
            data = agg.get('data', {})
            if 'avg_mse' in data:
                mse_values.append(data['avg_mse'])
            else:
                weights = np.array(data.get('weights', [0, 0]))
                mse_values.append(np.linalg.norm(weights))
            participants.append(data.get('num_participants', 0))
        
        fig.add_trace(
            go.Scatter(x=rounds, y=mse_values, mode='lines+markers+text',
                      name='MSE', line=dict(color='#E63946', width=3),
                      marker=dict(size=10, symbol='circle'),
                      text=[f'{val:.4f}' for val in mse_values],
                      textposition='top center',
                      hovertemplate='Round: %{x}<br>MSE: %{y:.6f}<extra></extra>'),
            row=1, col=1
        )
        
        # 2. Sensor Participation
        fig.add_trace(
            go.Bar(x=rounds, y=participants, name='Participants',
                   marker_color='#457B9D', opacity=0.8,
                   hovertemplate='Round: %{x}<br>Participants: %{y}<extra></extra>'),
            row=1, col=2
        )
        
        # 3. Local Training Progress (by sensor)
        if local_training:
            training_df = pd.DataFrame([{
                'sensor': lt.get('sender', 'unknown'),
                'round': lt.get('data', {}).get('federation_round', 0),
                'mse': lt.get('value', 0),
                'samples': lt.get('data', {}).get('num_samples', 0),
                'location': lt.get('data', {}).get('location', 'unknown')
            } for lt in local_training])
            
            for sensor in training_df['sensor'].unique():
                sensor_training_data = training_df[training_df['sensor'] == sensor]
                fig.add_trace(
                    go.Scatter(x=sensor_training_data['round'], y=sensor_training_data['mse'],
                              mode='lines+markers', name=f'{sensor}',
                              hovertemplate=f'{sensor}<br>Round: %{{x}}<br>MSE: %{{y:.6f}}<extra></extra>'),
                    row=2, col=1
                )
        
        # 4. Temperature Proposals Timeline
        if len(sensor_data) > 0:
            temp_df = pd.DataFrame([{
                'sensor': sd.get('sender', 'unknown'),
                'timestamp': sd.get('timestamp', ''),
                'temperature': sd.get('data', {}).get('temperature', 0),
                'proposal': sd.get('data', {}).get('proposal', 0),
                'confidence': sd.get('data', {}).get('confidence', 0)
            } for sd in sensor_data[:50]])  # Limit for performance
            
            fig.add_trace(
                go.Scatter(x=temp_df.index, y=temp_df['proposal'],
                          mode='markers', name='Temperature Proposals',
                          marker=dict(size=8, color=temp_df['confidence'], 
                                    colorscale='Viridis', showscale=True,
                                    colorbar=dict(title="Confidence", x=1.1)),
                          hovertemplate='Proposal: %{y:.2f}°C<br>Confidence: %{marker.color:.3f}<extra></extra>'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="🚀 Federated HVAC Learning Dashboard",
            title_font_size=20,
            showlegend=True,
            height=800,
            template='plotly_white'
        )
        
        # Save and show
        output_file = 'interactive_learning_dashboard.html'
        fig.write_html(output_file)
        print(f"✅ Interactive dashboard saved: {output_file}")
        fig.show()
    
    def create_interactive_hvac_timeline(self):
        """🏠 PLOTLY: Interactive HVAC Control Timeline."""
        metrics = self.metrics_list
        
        sensor_data = [m for m in metrics if m.get('type') == 'sensor_data']
        device_commands = [m for m in metrics if m.get('type') == 'device_command']
        
        if len(sensor_data) == 0 and len(device_commands) == 0:
            print("⚠️  No HVAC timeline data available")
            return
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('🌡️ Real-Time Temperature Readings', '⚙️ HVAC Mode Changes', '🎯 Setpoint Timeline'),
            shared_xaxes=True,
            vertical_spacing=0.08
        )
        
        # 1. Temperature readings by sensor location
        if len(sensor_data) > 0:
            temp_df = pd.DataFrame([{
                'timestamp': pd.to_datetime(sd.get('timestamp', '')),
                'sensor': sd.get('sender', 'unknown'),
                'location': sd.get('data', {}).get('location', 'unknown'),
                'temperature': sd.get('data', {}).get('temperature', 0),
                'proposal': sd.get('data', {}).get('proposal', 0),
                'confidence': sd.get('data', {}).get('confidence', 0)
            } for sd in sensor_data])
            
            locations = temp_df['location'].unique()
            colors = px.colors.qualitative.Set3
            
            for i, location in enumerate(locations):
                loc_data = temp_df[temp_df['location'] == location]
                fig.add_trace(
                    go.Scatter(x=loc_data['timestamp'], y=loc_data['temperature'],
                              mode='lines+markers', name=f'{location}',
                              line=dict(color=colors[i % len(colors)]),
                              hovertemplate=f'{location}<br>Time: %{{x}}<br>Temp: %{{y:.1f}}°C<extra></extra>'),
                    row=1, col=1
                )
        
        # 2. HVAC Mode timeline
        if len(device_commands) > 0:
            cmd_df = pd.DataFrame([{
                'timestamp': pd.to_datetime(dc.get('timestamp', '')),
                'old_mode': dc.get('data', {}).get('old_mode', 'UNKNOWN'),
                'new_mode': dc.get('data', {}).get('new_mode', 'UNKNOWN'),
                'setpoint': dc.get('data', {}).get('setpoint', 0)
            } for dc in device_commands])
            
            mode_colors = {'IDLE': '#A8DADC', 'HEAT': '#E63946', 'COOL': '#457B9D'}
            mode_values = {'IDLE': 0, 'HEAT': 2, 'COOL': 1}
            
            for mode in ['IDLE', 'HEAT', 'COOL']:
                mode_data = cmd_df[cmd_df['new_mode'] == mode]
                if not mode_data.empty:
                    fig.add_trace(
                        go.Scatter(x=mode_data['timestamp'], 
                                  y=[mode_values[mode]] * len(mode_data),
                                  mode='markers', name=f'{mode} Mode',
                                  marker=dict(color=mode_colors[mode], size=15, symbol='square'),
                                  hovertemplate=f'{mode}<br>Time: %{{x}}<br>Setpoint: {mode_data["setpoint"].iloc[0]:.1f}°C<extra></extra>'),
                        row=2, col=1
                    )
            
            # 3. Setpoint timeline
            fig.add_trace(
                go.Scatter(x=cmd_df['timestamp'], y=cmd_df['setpoint'],
                          mode='lines+markers', name='Setpoint',
                          line=dict(color='#F77F00', width=3),
                          marker=dict(size=8),
                          hovertemplate='Time: %{x}<br>Setpoint: %{y:.2f}°C<extra></extra>'),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title="🏠 Interactive HVAC Control Timeline",
            title_font_size=20,
            height=900,
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="HVAC Mode", row=2, col=1, 
                         tickvals=[0, 1, 2], ticktext=['IDLE', 'COOL', 'HEAT'])
        fig.update_yaxes(title_text="Setpoint (°C)", row=3, col=1)
        fig.update_xaxes(title_text="Time", row=3, col=1)
        
        # Save and show
        output_file = 'interactive_hvac_timeline.html'
        fig.write_html(output_file)
        print(f"✅ Interactive HVAC timeline saved: {output_file}")
        fig.show()
    
    def create_interactive_federation_analysis(self):
        """🔬 PLOTLY: Deep Federation Learning Analysis."""
        metrics = self.metrics_list
        
        aggregations = [m for m in metrics if m.get('type') == 'aggregation']
        local_training = [m for m in metrics if m.get('type') == 'local_training']
        
        if not aggregations or not local_training:
            print("⚠️  Insufficient data for federation analysis")
            return
        
        # Create comprehensive federation analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('📊 Model Weights Evolution', '⚖️ Weight Distribution by Round',
                          '🎯 Per-Sensor Learning Performance', '📈 Training Samples Distribution'),
            specs=[[{"secondary_y": False}, {"type": "box"}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        # 1. Model weights evolution
        rounds = []
        weight1_vals = []
        weight2_vals = []
        bias_vals = []
        
        for agg in aggregations:
            round_num = agg.get('round', 0)
            data = agg.get('data', {})
            weights = data.get('weights', [0, 0])
            bias = data.get('bias', 0)
            
            rounds.append(round_num)
            weight1_vals.append(weights[0] if len(weights) > 0 else 0)
            weight2_vals.append(weights[1] if len(weights) > 1 else 0)
            bias_vals.append(bias)
        
        fig.add_trace(
            go.Scatter(x=rounds, y=weight1_vals, mode='lines+markers',
                      name='Weight 1', line=dict(color='#E63946')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=rounds, y=weight2_vals, mode='lines+markers',
                      name='Weight 2', line=dict(color='#457B9D')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=rounds, y=bias_vals, mode='lines+markers',
                      name='Bias', line=dict(color='#F77F00')),
            row=1, col=1
        )
        
        # 2. Weight distribution box plot
        for round_num in rounds:
            fig.add_trace(
                go.Box(y=[weight1_vals[round_num-1], weight2_vals[round_num-1]], 
                       name=f'Round {round_num}',
                       boxpoints='all'),
                row=1, col=2
            )
        
        # 3. Per-sensor performance
        if len(local_training) > 0:
            sensor_performance = {}
            for lt in local_training:
                sensor = lt.get('sender', 'unknown')
                mse = lt.get('value', 0)
                round_num = lt.get('data', {}).get('federation_round', 0)
                
                if sensor not in sensor_performance:
                    sensor_performance[sensor] = {'rounds': [], 'mse': []}
                sensor_performance[sensor]['rounds'].append(round_num)
                sensor_performance[sensor]['mse'].append(mse)
            
            colors = px.colors.qualitative.Set1
            for i, (sensor, data) in enumerate(sensor_performance.items()):
                fig.add_trace(
                    go.Scatter(x=data['rounds'], y=data['mse'],
                              mode='lines+markers', name=sensor,
                              line=dict(color=colors[i % len(colors)])),
                    row=2, col=1
                )
        
        # 4. Training samples pie chart
        if len(local_training) > 0:
            sample_counts = {}
            for lt in local_training:
                sensor = lt.get('sender', 'unknown')
                samples = lt.get('data', {}).get('num_samples', 0)
                if sensor not in sample_counts:
                    sample_counts[sensor] = 0
                sample_counts[sensor] += samples
            
            fig.add_trace(
                go.Pie(labels=list(sample_counts.keys()), 
                       values=list(sample_counts.values()),
                       name="Training Samples"),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="🔬 Deep Federation Learning Analysis",
            title_font_size=20,
            height=900,
            showlegend=True,
            template='plotly_white'
        )
        
        # Save and show
        output_file = 'interactive_federation_analysis.html'
        fig.write_html(output_file)
        print(f"✅ Interactive federation analysis saved: {output_file}")
        fig.show()
    
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
        print("🚀 FEDERATED HVAC VISUALIZATION DASHBOARD".center(70))
        print("="*70 + "\n")
        
        if not self.load_data():
            return
        
        print("📊 Generating visualizations...\n")
        
        try:
            if self.interactive_mode:
                print("🎯 Creating interactive Plotly dashboards...")
                self.create_interactive_learning_dashboard()
                self.create_interactive_hvac_timeline()
                self.create_interactive_federation_analysis()
                print("\n📈 Creating static Matplotlib charts...")
            
            # Always generate static charts as backup
            self.plot_federation_convergence()
            self.plot_device_commands()
            self.plot_summary()
            
            print("\n" + "="*70)
            if self.interactive_mode:
                print("✅ Interactive + Static visualizations generated!".center(70))
                print("🌐 Open .html files in browser for interactive experience".center(70))
            else:
                print("✅ Static visualizations generated!".center(70))
            print("="*70 + "\n")
            
        except Exception as e:
            print(f"\n❌ Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
    
    def set_mode(self, interactive=True):
        """Toggle between interactive (Plotly) and static (Matplotlib) mode."""
        self.interactive_mode = interactive
        mode_name = "Interactive (Plotly)" if interactive else "Static (Matplotlib)"
        print(f"🔄 Visualization mode set to: {mode_name}")


def main():
    """Entry point."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Federated HVAC Visualization Dashboard')
    parser.add_argument('--log-file', default='logs/system_log.json', 
                       help='Path to log file (default: logs/system_log.json)')
    parser.add_argument('--static-only', action='store_true',
                       help='Generate only static matplotlib charts (no interactive Plotly)')
    parser.add_argument('--interactive-only', action='store_true',
                       help='Generate only interactive Plotly dashboards')
    
    args = parser.parse_args()
    
    dashboard = VisualizationDashboard(args.log_file)
    
    if args.static_only:
        dashboard.set_mode(interactive=False)
    elif args.interactive_only:
        dashboard.interactive_mode = True
        print("🎯 Generating interactive visualizations only...")
        if dashboard.load_data():
            dashboard.create_interactive_learning_dashboard()
            dashboard.create_interactive_hvac_timeline()  
            dashboard.create_interactive_federation_analysis()
        return
    
    dashboard.generate_all()


if __name__ == "__main__":
    main()
