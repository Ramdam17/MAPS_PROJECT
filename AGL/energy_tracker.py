import torch
import time
import psutil
import os
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

class NvidiaEnergyTracker:
    """Energy tracking class focused on NVIDIA GPU power measurements."""
    
    def __init__(self, project_name="gpu_training", output_dir="./energy_data", 
                 tracking_interval=5, country_iso_code="CAN"):
        """
        Initialize the energy and emissions tracker.
        
        Args:
            project_name: Name of the project for tracking purposes
            output_dir: Directory to save energy data
            tracking_interval: Interval (in seconds) to sample energy metrics
            country_iso_code: ISO code for the country (for emissions calculation)
        """
        self.project_name = project_name
        self.output_dir = output_dir
        self.tracking_interval = tracking_interval
        self.country_iso_code = country_iso_code
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if nvidia-smi is available
        self.has_nvidia_smi = self._check_nvidia_smi()
        if not self.has_nvidia_smi:
            raise RuntimeError("nvidia-smi not found. This tracker requires NVIDIA GPUs with nvidia-smi.")
        
        # Get number of GPUs
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if self.gpu_count == 0:
            raise RuntimeError("No NVIDIA GPUs detected. This tracker requires NVIDIA GPUs.")
            
        # Initialize tracking metrics
        self.start_time = None
        self.end_time = None
        self.gpu_readings = []  # List of dictionaries with GPU metrics
        self.time_readings = []
        self.cpu_util_readings = []
        self.is_tracking = False
        
        # Get GPU names
        self.gpu_names = self._get_gpu_names()
        print(f"Detected {self.gpu_count} GPUs: {', '.join(self.gpu_names)}")
        
        # Emissions factors for different countries (kg CO2 per kWh)
        # Source: https://www.carbonfootprint.com/international_electricity_factors.html
        self.emissions_factors = {
            "USA": 0.38,  # US average
            "CHN": 0.54,  # China
            "IND": 0.71,  # India
            "GBR": 0.23,  # UK
            "DEU": 0.34,  # Germany
            "CAN": 0.1,  # Canada
            "AUS": 0.78,  # Australia
            "FRA": 0.05,  # France
            "JPN": 0.42   # Japan
        }
        self.emissions_factor = self.emissions_factors.get(country_iso_code, 0.47)  # Default global average
        
    def _check_nvidia_smi(self):
        """Check if nvidia-smi is available."""
        try:
            subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def _get_gpu_names(self):
        """Get names of all available GPUs."""
        if not self.has_nvidia_smi:
            return []
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True, check=True
            )
            return [name.strip() for name in result.stdout.split('\n') if name.strip()]
        except subprocess.SubprocessError:
            return ["Unknown GPU"] * self.gpu_count
    
    def start(self):
        """Start tracking energy usage."""
        if self.is_tracking:
            print("Tracker is already running.")
            return
            
        # Record starting time
        self.start_time = time.time()
        
        # Reset readings
        self.gpu_readings = []
        self.time_readings = []
        self.cpu_util_readings = []
        
        # Take initial reading
        self._log_gpu_metrics()
        
        self.is_tracking = True
        print(f"Energy tracking started for project: {self.project_name}")
        print(f"Tracking {self.gpu_count} GPUs with nvidia-smi")
    
    def stop(self):
        """Stop tracking and return the results."""
        if not self.is_tracking:
            print("Tracker is not running.")
            return None
            
        # Record end time
        self.end_time = time.time()
        
        # Take final reading
        self._log_gpu_metrics()
        
        # Calculate energy usage
        total_energy_kwh = self._calculate_total_energy()
        
        # Calculate emissions
        emissions_kg = total_energy_kwh * self.emissions_factor
        
        # Total duration in hours
        duration = (self.end_time - self.start_time) / 3600  # Convert from seconds to hours
        
        # Calculate metrics from readings
        power_metrics = self._calculate_power_metrics()
        
        self.is_tracking = False
        
        # Create a summary of the tracking session
        result = {
            "project_name": self.project_name,
            "duration_hours": duration,
            "energy_kwh": total_energy_kwh,
            "emissions_kg": emissions_kg,
            "start_time": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.fromtimestamp(self.end_time).strftime("%Y-%m-%d %H:%M:%S"),
            "power_metrics": power_metrics
        }
        
        # Save detailed logs
        self._save_logs()
        
        print(f"Energy used: {total_energy_kwh:.6f} kWh")
        print(f"Carbon emissions: {emissions_kg:.6f} kg CO2eq")
        print(f"Duration: {duration:.2f} hours")
        
        return result
    
    def log_point(self):
        """Log current GPU metrics."""
        if not self.is_tracking:
            print("Tracker is not running.")
            return
        
        self._log_gpu_metrics()
    
    def _log_gpu_metrics(self):
        """Get and log GPU metrics using nvidia-smi."""
        if not self.has_nvidia_smi:
            return
        
        current_time = time.time()
        self.time_readings.append(current_time - self.start_time if self.start_time else 0)
        
        # Get CPU utilization
        cpu_util = psutil.cpu_percent()
        self.cpu_util_readings.append(cpu_util)
        
        # Get GPU metrics for all GPUs
        gpu_data = {}
        
        # Get power usage
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,power.draw,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True
            )
            
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                    
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 7:  # Ensure we have all expected metrics
                    idx = int(parts[0])
                    gpu_data[idx] = {
                        'power_watts': float(parts[1]),
                        'gpu_util': float(parts[2]),
                        'mem_util': float(parts[3]),
                        'mem_used_mb': float(parts[4]),
                        'mem_total_mb': float(parts[5]),
                        'temp_c': float(parts[6])
                    }
        except (subprocess.SubprocessError, ValueError, IndexError) as e:
            print(f"Error getting GPU metrics: {e}")
            # Create empty data if we couldn't get metrics
            for i in range(self.gpu_count):
                gpu_data[i] = {
                    'power_watts': 0,
                    'gpu_util': 0,
                    'mem_util': 0,
                    'mem_used_mb': 0,
                    'mem_total_mb': 0,
                    'temp_c': 0
                }
        
        self.gpu_readings.append(gpu_data)
    
    def _calculate_total_energy(self):
        """Calculate total energy used in kWh based on power readings."""
        if len(self.time_readings) < 2:
            return 0
        
        total_energy_joules = 0
        
        # Calculate energy (power * time) for each interval
        for i in range(1, len(self.time_readings)):
            time_diff = self.time_readings[i] - self.time_readings[i-1]  # Time difference in seconds
            
            # Sum power from all GPUs for the previous reading
            # (using trapezoid rule for better accuracy - average of start and end power)
            power_start = sum(gpu.get('power_watts', 0) for gpu in self.gpu_readings[i-1].values())
            power_end = sum(gpu.get('power_watts', 0) for gpu in self.gpu_readings[i].values())
            avg_power = (power_start + power_end) / 2
            
            # Energy in joules = power (watts) * time (seconds)
            energy_joules = avg_power * time_diff
            total_energy_joules += energy_joules
        
        # Convert joules to kWh
        return total_energy_joules / 3600000  # 1 kWh = 3,600,000 joules
    
    def _calculate_power_metrics(self):
        """Calculate power and utilization metrics from readings."""
        if not self.gpu_readings:
            return {
                "avg_power_watts": 0,
                "peak_power_watts": 0,
                "avg_gpu_util": 0,
                "peak_gpu_util": 0,
                "avg_gpu_temp": 0,
                "peak_gpu_temp": 0,
                "per_gpu_metrics": {}
            }
        
        # Initialize metrics
        total_power_readings = []
        total_util_readings = []
        total_temp_readings = []
        per_gpu_metrics = {}
        
        # Process all readings
        for reading in self.gpu_readings:
            # Sum power across all GPUs for this reading
            total_power = sum(gpu.get('power_watts', 0) for gpu in reading.values())
            total_power_readings.append(total_power)
            
            # Average utilization across all GPUs
            utils = [gpu.get('gpu_util', 0) for gpu in reading.values()]
            if utils:
                total_util_readings.append(sum(utils) / len(utils))
            
            # Average temperature across all GPUs
            temps = [gpu.get('temp_c', 0) for gpu in reading.values()]
            if temps:
                total_temp_readings.append(sum(temps) / len(temps))
            
            # Collect per-GPU metrics
            for idx, gpu_data in reading.items():
                if idx not in per_gpu_metrics:
                    per_gpu_metrics[idx] = {
                        'power_readings': [],
                        'util_readings': [],
                        'temp_readings': []
                    }
                
                per_gpu_metrics[idx]['power_readings'].append(gpu_data.get('power_watts', 0))
                per_gpu_metrics[idx]['util_readings'].append(gpu_data.get('gpu_util', 0))
                per_gpu_metrics[idx]['temp_readings'].append(gpu_data.get('temp_c', 0))
        
        # Calculate aggregate metrics
        result = {
            "avg_power_watts": np.mean(total_power_readings) if total_power_readings else 0,
            "peak_power_watts": np.max(total_power_readings) if total_power_readings else 0,
            "avg_gpu_util": np.mean(total_util_readings) if total_util_readings else 0,
            "peak_gpu_util": np.max(total_util_readings) if total_util_readings else 0,
            "avg_gpu_temp": np.mean(total_temp_readings) if total_temp_readings else 0,
            "peak_gpu_temp": np.max(total_temp_readings) if total_temp_readings else 0,
            "per_gpu_metrics": {}
        }
        
        # Calculate per-GPU summary metrics
        for idx, metrics in per_gpu_metrics.items():
            result["per_gpu_metrics"][idx] = {
                "avg_power_watts": np.mean(metrics['power_readings']) if metrics['power_readings'] else 0,
                "peak_power_watts": np.max(metrics['power_readings']) if metrics['power_readings'] else 0,
                "avg_util": np.mean(metrics['util_readings']) if metrics['util_readings'] else 0,
                "peak_util": np.max(metrics['util_readings']) if metrics['util_readings'] else 0,
                "avg_temp": np.mean(metrics['temp_readings']) if metrics['temp_readings'] else 0,
                "peak_temp": np.max(metrics['temp_readings']) if metrics['temp_readings'] else 0
            }
        
        return result
    
    def _save_logs(self):
        """Save detailed logs to CSV and generate plots."""
        if not self.gpu_readings:
            print("No data points were collected during tracking.")
            return
            
        # Create DataFrame for time series data
        time_data = {'time_seconds': self.time_readings, 'cpu_util': self.cpu_util_readings}
        
        # Add columns for each GPU's metrics
        for gpu_idx in range(self.gpu_count):
            # Initialize with zeros in case some readings are missing
            time_data[f'gpu{gpu_idx}_power'] = [0] * len(self.time_readings)
            time_data[f'gpu{gpu_idx}_util'] = [0] * len(self.time_readings)
            time_data[f'gpu{gpu_idx}_temp'] = [0] * len(self.time_readings)
        
        # Fill in actual readings
        for i, reading in enumerate(self.gpu_readings):
            for gpu_idx, gpu_data in reading.items():
                time_data[f'gpu{gpu_idx}_power'][i] = gpu_data.get('power_watts', 0)
                time_data[f'gpu{gpu_idx}_util'][i] = gpu_data.get('gpu_util', 0)
                time_data[f'gpu{gpu_idx}_temp'][i] = gpu_data.get('temp_c', 0)
        
        # Create DataFrame
        df = pd.DataFrame(time_data)
        
        # Add total power column
        power_cols = [col for col in df.columns if col.endswith('_power')]
        df['total_power'] = df[power_cols].sum(axis=1)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.output_dir, f"{self.project_name}_energy_log_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        
        # Generate plots
        self._generate_plots(df, timestamp)
        
        print(f"Detailed logs saved to {csv_path}")
    
    def _generate_plots(self, df, timestamp):
        """Generate plots of energy usage and system utilization."""
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot 1: Power usage over time
        power_cols = [col for col in df.columns if col.endswith('_power')]
        for col in power_cols:
            if col == 'total_power':
                axes[0].plot(df['time_seconds']/60, df[col], 'k-', linewidth=2, label='Total')
            else:
                gpu_idx = col.split('_')[0][3:]  # Extract index from "gpuX_power"
                axes[0].plot(df['time_seconds']/60, df[col], '-', label=f'GPU {gpu_idx}')
        
        axes[0].set_title('GPU Power Usage Over Time')
        axes[0].set_xlabel('Time (minutes)')
        axes[0].set_ylabel('Power (Watts)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot 2: GPU Utilization
        util_cols = [col for col in df.columns if col.endswith('_util') and col.startswith('gpu')]
        for col in util_cols:
            gpu_idx = col.split('_')[0][3:]  # Extract index from "gpuX_util"
            axes[1].plot(df['time_seconds']/60, df[col], '-', label=f'GPU {gpu_idx}')
        
        axes[1].set_title('GPU Utilization Over Time')
        axes[1].set_xlabel('Time (minutes)')
        axes[1].set_ylabel('Utilization (%)')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot 3: GPU Temperature
        temp_cols = [col for col in df.columns if col.endswith('_temp')]
        for col in temp_cols:
            gpu_idx = col.split('_')[0][3:]  # Extract index from "gpuX_temp"
            axes[2].plot(df['time_seconds']/60, df[col], '-', label=f'GPU {gpu_idx}')
        
        axes[2].set_title('GPU Temperature Over Time')
        axes[2].set_xlabel('Time (minutes)')
        axes[2].set_ylabel('Temperature (°C)')
        axes[2].legend()
        axes[2].grid(True)
        
        # Save figure
        plot_path = os.path.join(self.output_dir, f"{self.project_name}_energy_plot_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Plots saved to {plot_path}")


class MLModelEnergyEfficiency:
    """
    Class to calculate and report energy efficiency metrics for ML models.
    """
    
    def __init__(self, energy_tracker, model_name="unnamed_model"):
        """
        Initialize with an energy tracker and model name.
        
        Args:
            energy_tracker: Instance of NvidiaEnergyTracker
            model_name: Name of the model being evaluated
        """
        self.energy_tracker = energy_tracker
        self.model_name = model_name
        
    def start_tracking(self):
        """Start energy tracking for the model."""
        self.energy_tracker.start()
        
    def stop_tracking(self):
        """Stop energy tracking and return results."""
        return self.energy_tracker.stop()
    
    def calculate_efficiency_metrics(self, training_results, energy_results=None):
        """
        Calculate efficiency metrics for the model.
        
        Args:
            training_results: Dictionary containing model performance metrics
            energy_results: Energy tracking results (optional if already tracking)
        
        Returns:
            Dictionary of efficiency metrics
        """
        # If no energy results provided, use the last tracking results
        if energy_results is None:
            energy_results = self.energy_tracker.stop()
            
        energy_kwh = energy_results['energy_kwh']
        emissions_kg = energy_results['emissions_kg']
        
        metrics = {
            "model_name": self.model_name,
            "energy_kwh": energy_kwh,
            "emissions_kg": emissions_kg,
            "training_time_hours": energy_results['duration_hours']
        }
        
        # Calculate efficiency metrics based on available training results
        if 'accuracy' in training_results:
            metrics['kwh_per_percent_accuracy'] = energy_kwh / training_results['accuracy']
            metrics['emissions_per_percent_accuracy'] = emissions_kg / training_results['accuracy']
            metrics['accuracy_per_kwh'] = training_results['accuracy'] / energy_kwh
        
        if 'loss' in training_results:
            # Lower loss is better, so invert for efficiency
            metrics['loss_efficiency'] = 1 / (energy_kwh * training_results['loss'])
        
        if 'model_parameters' in training_results:
            params_millions = training_results['model_parameters'] / 1_000_000
            metrics['kwh_per_million_params'] = energy_kwh / params_millions
            metrics['emissions_per_million_params'] = emissions_kg / params_millions
        
        if 'inference_time_ms' in training_results:
            # Energy efficiency for inference
            metrics['energy_time_product'] = energy_kwh * training_results['inference_time_ms']
        
        if 'epochs' in training_results:
            metrics['kwh_per_epoch'] = energy_kwh / training_results['epochs']
            metrics['emissions_per_epoch'] = emissions_kg / training_results['epochs']
        
        return metrics
    
    def generate_efficiency_report(self, efficiency_metrics, output_dir=None):
        """
        Generate a detailed report on model energy efficiency.
        
        Args:
            efficiency_metrics: Dictionary of efficiency metrics
            output_dir: Directory to save the report (default: same as energy tracker)
        
        Returns:
            Path to the saved report
        """
        if output_dir is None:
            output_dir = self.energy_tracker.output_dir
        
        # Create report content
        report = [
            f"# Energy Efficiency Report for {self.model_name}",
            "",
            "## Overview",
            f"- **Model Name**: {self.model_name}",
            f"- **Total Energy**: {efficiency_metrics['energy_kwh']:.6f} kWh",
            f"- **Carbon Emissions**: {efficiency_metrics['emissions_kg']:.6f} kg CO2eq",
            f"- **Training Time**: {efficiency_metrics['training_time_hours']:.2f} hours",
            "",
            "## Efficiency Metrics"
        ]
        
        # Add metrics sections
        if 'accuracy' in efficiency_metrics:
            report.extend([
                "",
                "### Accuracy Metrics",
                f"- **kWh per % Accuracy**: {efficiency_metrics['kwh_per_percent_accuracy']:.6f}",
                f"- **Accuracy per kWh**: {efficiency_metrics['accuracy_per_kwh']:.6f}"
            ])
        
        if 'loss' in efficiency_metrics:
            report.extend([
                "",
                "### Loss Metrics",
                f"- **Loss Efficiency**: {efficiency_metrics['loss_efficiency']:.6f}"
            ])
        
        if 'model_parameters' in efficiency_metrics:
            report.extend([
                "",
                "### Model Size Metrics",
                f"- **kWh per Million Parameters**: {efficiency_metrics['kwh_per_million_params']:.6f}",
                f"- **Emissions per Million Parameters**: {efficiency_metrics['emissions_per_million_params']:.6f}"
            ])
        
        if 'epochs' in efficiency_metrics:
            report.extend([
                "",
                "### Training Metrics",
                f"- **kWh per Epoch**: {efficiency_metrics['kwh_per_epoch']:.6f}",
                f"- **Emissions per Epoch**: {efficiency_metrics['emissions_per_epoch']:.6f}"
            ])
        
        # Add recommendations
        report.extend([
            "",
            "## Recommendations",
            "Based on the energy efficiency metrics, consider the following optimizations:",
            "",
            "1. **Hardware Utilization**: Ensure GPUs are properly utilized during training",
            "2. **Batch Size Optimization**: Find the optimal batch size for energy efficiency",
            "3. **Model Architecture**: Consider more energy-efficient architectures",
            "4. **Early Stopping**: Implement early stopping to avoid unnecessary training",
            "5. **Quantization**: Consider quantized models for inference",
            "",
            f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ])
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"{self.model_name}_efficiency_report_{timestamp}.md")
        
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Efficiency report saved to {report_path}")
        return report_path

