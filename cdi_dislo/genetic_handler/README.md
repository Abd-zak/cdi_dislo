# CDI Reconstruction Utilities

This submodule offers a collection of essential utilities for Coherent Diffraction Imaging (CDI) reconstructions, facilitating multiple reconstructions, object ranking based on sharpness, genetic optimization approaches, and data preprocessing techniques.

---

## Overview

This module provides the following key functionalities:

- Multiple CDI reconstructions with robust error handling.
- Sharpness-based object sorting for automated ranking and selection.
- Genetic algorithm-driven refinement of reconstructions.
- Automatic cropping around the center of mass for optimized visualization and analysis.

---

## Key Functionalities

### Reconstruction Utilities
- `make_several_reconstruction()`: Executes multiple CDI reconstructions with automated error handling.

### Sharpness and Sorting
- `sharpness_metric()`: Computes sharpness scores for object quality assessment.
- `sort_objects()`: Sorts reconstructed objects by sharpness, optionally providing visual feedback.

### Genetic Optimization
- `genetic_update_object_list()`: Refines object lists through genetic-inspired operations, ensuring complex conjugate consistency.

### Automatic Data Cropping
- `automatic_crop()`: Determines optimal cropping dimensions centered around the object's mass, ensuring zero-value cropping is avoided.

---

## Dependencies

Core dependencies:
- NumPy
- Matplotlib
- `cdi_dislo.ewen_utilities` (internal utilities for reconstruction, plotting, and post-processing)

Installation via pip:
```bash
pip install numpy matplotlib
```

*(Note: Internal utilities require the complete `cdi_dislo` package.)*

---

## Usage Example

Basic usage example:

```python
from cdi_dislo.cdi_reconstruction import make_several_reconstruction, sort_objects

# Perform multiple reconstructions
objects, supports = make_several_reconstruction(data, nb_recon=10, params_init=initial_params)

# Sort reconstructed objects by sharpness
metrics, sorted_objects, sorted_supports = sort_objects(objects, supports, plot=True)
```

---

## Suggestions for Future Enhancements

- **Enhanced Error Handling:** Replace broad exceptions with specific error catching and structured logging.
- **Performance Optimization:** Implement parallel processing and NumPy vectorization for speed enhancements.
- **Advanced Genetic Algorithms:** Incorporate mutations, adaptive parameter tuning, and crossover techniques.
- **Modularity and Testing:** Increase modularity, implement thorough unit tests, and add comprehensive input validation.
- **Visualization and UX:** Introduce progress bars, detailed logging modes, and enriched plotting capabilities.

---

## Testing

Run tests with pytest:
```bash
pytest tests/test_cdi_reconstruction.py
```

---

## License

Distributed under the MIT License. See [LICENSE](../../LICENSE) for more details.

