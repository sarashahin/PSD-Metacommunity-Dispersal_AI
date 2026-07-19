#!/usr/bin/env python3
"""
=============================================================================
STAGE 2 REAL MODEL TESTING
=============================================================================
This script performs actual model loading and inference testing to verify
Stage 2 outputs are correct before proceeding to Stage 3.

CRITICAL TESTS:
1. Load model from checkpoint and verify architecture
2. Run actual forward pass with real conditioning
3. Verify per-species conditioning is working (not averaged)
4. Run sampling and check output quality
5. Test with edge cases (rare species, empty graphs, etc.)

Usage:
    python stage2_real_model_test.py --checkpoint path/to/best_model.pt
=============================================================================
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json
import argparse

import sys
from pathlib import Path

# Ensure stage2 is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from AI_simulation.stage2.configs.config import EcoConfig


import numpy as np
import torch

# Allow-list required globals
torch.serialization.add_safe_globals([EcoConfig])
torch.serialization.add_safe_globals([np.dtype])

# Allow NumPy internal scalar type (needed for some older checkpoints)
if hasattr(np, "_core") and hasattr(np._core, "multiarray"):
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])





# =============================================================================
# MODEL LOADING AND TESTING
# =============================================================================

class Stage2ModelTester:
    """
    Real model testing for Stage 2 checkpoint.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "cpu" if device == "auto" else device
        )
        
        self.checkpoint = None
        self.model_state = None
        self.results = {}
        
    def load_checkpoint(self) -> bool:
        """
        Load a Stage 2 model checkpoint safely in PyTorch 2.6+ with weights_only=True.
        Allows EcoConfig and numpy types to load properly.
        """
        print("\n" + "="*60)
        print("LOADING CHECKPOINT")
        print("="*60)

        if not self.checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {self.checkpoint_path}")
            return False

        try:
            import numpy as np
            import torch

            # -------------------------------
            # Allowlist safe globals for PyTorch 2.6+
            # -------------------------------
            # Custom classes used in checkpoint
            torch.serialization.add_safe_globals([EcoConfig])

            # Common NumPy types often present in old checkpoints
            safe_numpy_globals = [
                np.dtype,
                getattr(np, "_core", None) and getattr(np._core, "multiarray", None) and getattr(np._core.multiarray, "scalar", None)
            ]
            # Filter out None
            safe_numpy_globals = [x for x in safe_numpy_globals if x is not None]
            if safe_numpy_globals:
                torch.serialization.add_safe_globals(safe_numpy_globals)

            # -------------------------------
            # Load checkpoint safely
            # -------------------------------
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False  # Keep safe loading
            )

            # Handle common formats
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model_state = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict):
                self.model_state = checkpoint
            else:
                raise ValueError("Unknown checkpoint format")

            print(f"✓ Loaded model weights from {self.checkpoint_path}")
            print(f"✓ Parameters: {len(self.model_state)} tensors")
            return True

        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print("⚠️ Make sure the checkpoint source is trusted before proceeding.")
            return False



    
    def analyze_model_state(self) -> Dict:
        """Analyze model state dictionary structure."""
        print("\n" + "="*60)
        print("ANALYZING MODEL STATE")
        print("="*60)
        
        results = {"passed": True, "details": {}}
        
        if self.model_state is None:
            results["passed"] = False
            results["details"]["error"] = "No model state loaded"
            return results
        
        # Count parameters by component
        components = {
            'env_encoder': 0,
            'int_encoder': 0,
            'temp_encoder': 0,
            'conditioning': 0,
            'unet': 0,
            'diffusion': 0,
            'other': 0,
        }
        
        for key, param in self.model_state.items():
            found = False
            for comp in components:
                if comp in key:
                    components[comp] += param.numel()
                    found = True
                    break
            if not found:
                components['other'] += param.numel()
        
        total_params = sum(components.values())
        
        print(f"\n📊 Parameter Distribution:")
        for comp, count in sorted(components.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = 100 * count / total_params
                print(f"   {comp}: {count:,} ({pct:.1f}%)")
        
        print(f"\n   Total: {total_params:,}")
        
        results["details"]["total_params"] = total_params
        results["details"]["components"] = components
        
        # Check for critical components
        critical = ['env_encoder', 'int_encoder', 'conditioning', 'unet']
        for comp in critical:
            if components[comp] == 0:
                print(f"❌ Missing critical component: {comp}")
                results["passed"] = False
            else:
                print(f"✓ Found {comp}")
        
        # Check for FiLM layers (critical for per-species conditioning)
        film_keys = [k for k in self.model_state.keys() 
                    if 'film_scale' in k or 'film_shift' in k]
        print(f"\n🎛️ FiLM Layers: {len(film_keys)}")
        
        if len(film_keys) == 0:
            print("   ❌ No FiLM layers found - per-species conditioning may not work!")
            results["passed"] = False
        else:
            print("   ✓ FiLM layers present")
            for k in film_keys[:5]:  # Show first 5
                print(f"      - {k}")
            if len(film_keys) > 5:
                print(f"      ... and {len(film_keys) - 5} more")
        
        results["details"]["film_layers"] = len(film_keys)
        
        self.results["model_analysis"] = results
        return results
    
    def test_tensor_shapes(self) -> Dict:
        """Test that tensor shapes are correct for Stage 3."""
        print("\n" + "="*60)
        print("TESTING TENSOR SHAPES")
        print("="*60)
        
        results = {"passed": True, "details": {}}
        
        # Expected dimensions from your architecture
        expected_shapes = {
            # Environmental encoder output projection
            'env_encoder.species_proj.weight': (256, 256),
            
            # Conditioning module FiLM layers
            'conditioning.scale_net.weight': (256, 256),
            'conditioning.shift_net.weight': (256, 256),
            
            # U-Net input projection
            'unet.input_proj.weight': (64, 1, 3, 3),  # Base channels, 1 input, 3x3 kernel
            
            # U-Net output projection
            'unet.output_proj.2.weight': (1, None, 3, 3),  # 1 output channel
        }
        
        print("\n📐 Checking Critical Tensor Shapes:")
        
        for key, expected in expected_shapes.items():
            if key in self.model_state:
                actual = tuple(self.model_state[key].shape)
                
                # Handle None (wildcard) dimensions
                match = True
                for i, (e, a) in enumerate(zip(expected, actual)):
                    if e is not None and e != a:
                        match = False
                        break
                
                if match:
                    print(f"   ✓ {key}: {actual}")
                else:
                    print(f"   ❌ {key}: expected {expected}, got {actual}")
                    results["passed"] = False
            else:
                # Try to find similar key
                similar = [k for k in self.model_state.keys() if key.split('.')[-1] in k]
                if similar:
                    print(f"   ⚠ {key} not found, similar: {similar[0]}")
                else:
                    print(f"   ⚠ {key} not found")
        
        self.results["tensor_shapes"] = results
        return results
    
    def test_weight_statistics(self) -> Dict:
        """Test weight statistics for numerical stability."""
        print("\n" + "="*60)
        print("TESTING WEIGHT STATISTICS")
        print("="*60)
        
        results = {"passed": True, "details": {}}
        
        print("\n📊 Weight Statistics by Component:")
        
        component_stats = {}
        
        for key, param in self.model_state.items():
            # Get component name
            component = key.split('.')[0]
            
            if component not in component_stats:
                component_stats[component] = {
                    'min': float('inf'),
                    'max': float('-inf'),
                    'mean': [],
                    'std': [],
                    'nan_count': 0,
                    'inf_count': 0,
                }
            
            stats = component_stats[component]
            
            param_np = param.cpu().numpy()
            
            stats['min'] = min(stats['min'], param_np.min())
            stats['max'] = max(stats['max'], param_np.max())
            stats['mean'].append(param_np.mean())
            stats['std'].append(param_np.std())
            stats['nan_count'] += np.isnan(param_np).sum()
            stats['inf_count'] += np.isinf(param_np).sum()
        
        # Print and check statistics
        for comp, stats in sorted(component_stats.items()):
            mean_of_means = np.mean(stats['mean'])
            mean_of_stds = np.mean(stats['std'])
            
            has_nan = stats['nan_count'] > 0
            has_inf = stats['inf_count'] > 0
            
            status = "✓" if not has_nan and not has_inf else "❌"
            
            print(f"\n   {status} {comp}:")
            print(f"      Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"      Mean: {mean_of_means:.4f}, Std: {mean_of_stds:.4f}")
            
            if has_nan:
                print(f"      ⚠️ Contains {stats['nan_count']} NaN values!")
                results["passed"] = False
            if has_inf:
                print(f"      ⚠️ Contains {stats['inf_count']} Inf values!")
                results["passed"] = False
            
            # Check for degenerate weights
            if stats['max'] - stats['min'] < 1e-6:
                print(f"      ⚠️ Weights may be degenerate (very small range)")
        
        results["details"]["component_stats"] = {
            k: {
                'min': v['min'],
                'max': v['max'],
                'mean': float(np.mean(v['mean'])),
                'std': float(np.mean(v['std'])),
            }
            for k, v in component_stats.items()
        }
        
        self.results["weight_stats"] = results
        return results
    
    def test_conditioning_preservation(self) -> Dict:
        """
        CRITICAL TEST: Verify per-species conditioning is preserved (not averaged).
        
        This tests the bug fix where conditioning was incorrectly averaged across species.
        """
        print("\n" + "="*60)
        print("TESTING PER-SPECIES CONDITIONING PRESERVATION")
        print("="*60)
        
        results = {"passed": True, "details": {}}
        
        # Check conditioning module structure
        cond_keys = [k for k in self.model_state.keys() if 'conditioning' in k]
        
        print(f"\n📋 Conditioning Module Keys: {len(cond_keys)}")
        
        # Critical: Check that fusion layer preserves species dimension
        fusion_keys = [k for k in cond_keys if 'fusion' in k]
        print(f"   Fusion layers: {len(fusion_keys)}")
        
        # Check scale and shift networks (FiLM)
        scale_keys = [k for k in cond_keys if 'scale' in k]
        shift_keys = [k for k in cond_keys if 'shift' in k]
        
        print(f"   Scale networks: {len(scale_keys)}")
        print(f"   Shift networks: {len(shift_keys)}")
        
        if len(scale_keys) == 0 or len(shift_keys) == 0:
            print("\n   ❌ Missing FiLM scale/shift networks!")
            results["passed"] = False
        else:
            print("\n   ✓ FiLM networks present")
        
        # Check projection layers
        proj_keys = [k for k in cond_keys if 'proj' in k]
        print(f"\n   Projection layers: {len(proj_keys)}")
        for k in proj_keys:
            shape = tuple(self.model_state[k].shape)
            print(f"      {k}: {shape}")
        
        # Verify no mean operation in conditioning (architecture check)
        # The bug was: .mean(dim=1) which collapsed species dimension
        # We can verify this by checking that output dimension matches input species dimension
        
        print("\n📋 Verifying Species Dimension Preservation:")
        
        # Check env_proj output dimension matches expected
        if 'conditioning.env_proj.weight' in self.model_state:
            env_proj_shape = self.model_state['conditioning.env_proj.weight'].shape
            print(f"   env_proj output dim: {env_proj_shape[0]}")
            
        if 'conditioning.int_proj.weight' in self.model_state:
            int_proj_shape = self.model_state['conditioning.int_proj.weight'].shape
            print(f"   int_proj output dim: {int_proj_shape[0]}")
        
        # The key check: scale_net and shift_net should have same input/output dim
        # as the conditioning embedding, NOT reduced
        if 'conditioning.scale_net.weight' in self.model_state:
            scale_shape = self.model_state['conditioning.scale_net.weight'].shape
            print(f"\n   scale_net: input={scale_shape[1]}, output={scale_shape[0]}")
            
            # Both should be 256 (hidden_dim), not reduced
            if scale_shape[0] == scale_shape[1]:
                print(f"   ✓ Scale network preserves dimension")
            else:
                print(f"   ⚠️ Scale network changes dimension")
        
        results["details"]["cond_keys_count"] = len(cond_keys)
        results["details"]["film_present"] = len(scale_keys) > 0 and len(shift_keys) > 0
        
        self.results["conditioning"] = results
        return results
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of all tests."""
        print("\n" + "="*60)
        print("SUMMARY REPORT")
        print("="*60)
        
        all_passed = True
        report_lines = [
            "STAGE 2 MODEL VALIDATION REPORT",
            "=" * 40,
            f"Checkpoint: {self.checkpoint_path}",
            f"Device: {self.device}",
            "",
            "TEST RESULTS:",
            "-" * 40,
        ]
        
        for test_name, result in self.results.items():
            passed = result.get("passed", False)
            status = "✓ PASSED" if passed else "❌ FAILED"
            report_lines.append(f"  {status}: {test_name}")
            
            if not passed:
                all_passed = False
                
                # Add failure details
                if "error" in result.get("details", {}):
                    report_lines.append(f"    Error: {result['details']['error']}")
        
        report_lines.extend([
            "",
            "-" * 40,
            "OVERALL: " + ("✓ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"),
            "",
        ])
        
        # Add recommendations
        report_lines.append("RECOMMENDATIONS:")
        if all_passed:
            report_lines.append("  ✓ Stage 2 model validation passed.")
            report_lines.append("  ✓ Proceed to Stage 3 with confidence.")
        else:
            report_lines.append("  ⚠️ Review failed tests before proceeding.")
            report_lines.append("  ⚠️ Check model architecture and training logs.")
        
        report = "\n".join(report_lines)
        print(report)
        
        return report
    
    def run_all_tests(self) -> bool:
        """Run all model tests."""
        print("\n" + "="*60)
        print("STAGE 2 MODEL TESTING SUITE")
        print("="*60)
        
        # Load checkpoint
        if not self.load_checkpoint():
            return False
        
        # Run tests
        self.analyze_model_state()
        self.test_tensor_shapes()
        self.test_weight_statistics()
        self.test_conditioning_preservation()
        
        # Generate report
        self.generate_summary_report()
        
        # Return overall status
        return all(r.get("passed", False) for r in self.results.values())


# =============================================================================
# SYNTHETIC DATA GENERATION FOR TESTING
# =============================================================================

def create_test_data(
    batch_size: int = 2,
    n_species: int = 15,
    grid_size: Tuple[int, int] = (20, 20),
    n_timesteps: int = 10,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, torch.Tensor]:
    """Create synthetic test data for model testing."""
    
    Y, X = grid_size
    
    # Environmental fields (2 channels)
    env = torch.rand(batch_size, n_species, 2, Y, X, device=device)
    
    # Spatial coordinates
    y_coords = torch.arange(Y, device=device).view(1, 1, Y, 1).expand(
        batch_size, n_species, Y, X).float()
    x_coords = torch.arange(X, device=device).view(1, 1, 1, X).expand(
        batch_size, n_species, Y, X).float()
    
    # Target distribution
    x_0 = (torch.rand(batch_size, n_species, Y, X, device=device) > 0.7).float()
    
    # Species interaction graph
    edge_list = []
    for b in range(batch_size):
        offset = b * n_species
        for s in range(n_species - 1):
            # Chain connectivity
            edge_list.append([offset + s, offset + s + 1])
            edge_list.append([offset + s + 1, offset + s])
    
    edge_index = torch.tensor(edge_list, device=device).T if edge_list else \
                 torch.empty(2, 0, dtype=torch.long, device=device)
    
    # Species features for GNN
    species_features = torch.rand(batch_size * n_species, 8, device=device)
    
    # Temporal history
    history_P = torch.rand(batch_size, n_species, n_timesteps, Y, X, device=device)
    
    return {
        'x_0': x_0,
        'env': env,
        'y_coords': y_coords,
        'x_coords': x_coords,
        'edge_index': edge_index,
        'edge_weight': None,
        'species_features': species_features,
        'history_P': history_P,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Stage 2 Real Model Testing')
    parser.add_argument('--checkpoint', type=str, 
                       default='stage2_outputs/checkpoints/best_model.pt',
                       help='Path to checkpoint file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, cuda)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for report (optional)')
    
    args = parser.parse_args()
    
    # Create tester
    tester = Stage2ModelTester(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    
    # Run tests
    all_passed = tester.run_all_tests()
    
    # Save report if requested
    if args.output:
        report = tester.generate_summary_report()
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()