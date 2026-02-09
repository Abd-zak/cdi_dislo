# Reconstruction File Selection Submodule

This submodule automates the selection and post-processing of reconstructed 3D phase and density data from Coherent Diffraction Imaging (CDI) experiments. It selects reconstructions based on specific metrics such as Log-Likelihood factor (LLKf), standard deviation of density (stdrho), and number of active pixels.

---

## Overview

This script allows:

- Automated or manual selection of reconstructions based on quantitative criteria.
- Visualization of selection criteria for easier interpretation.
- Creation of mode files (`modes.h5`) from selected reconstructions.
- Optional archiving and cleanup of previous selections.
- Automated plotting and saving of detailed logs and visualizations.

---

## Key Functionalities

### Selection Methods
- **Automatic (`auto`)**: Selects based on default thresholds.
- **Interactive Loop (`looped`)**: Allows dynamic user interaction for refining selections.
- **Filter by LLKf (`filter_by_llkf`)**: Selects based solely on LLKf values.
- **Manual (`manual`)**: Fully user-driven selection.

### Processing and Outputs
- Loads and processes `.CXI` files.
- Saves selected reconstructions and generates logs.
- Provides detailed visualizations including:
  - LLKf vs. Run plots
  - Standard deviation of density distributions
  - Phase animations and 3D projections

### Utilities and Post-processing
- Optionally saves animations showing phase evolution.
- Creates mode files for further analysis using `pynx-cdi-analysis`.

---

## Dependencies

Main dependencies:
- NumPy
- Matplotlib
- seaborn
- pynx-cdi-analysis (for mode file generation)

Installation:
```bash
pip install numpy matplotlib seaborn
```

*(Note: `pynx-cdi-analysis` should be installed separately.)*

---

## Usage Example

```python
from cdi_dislo.selection_utilities import selection_of_reco_in_path

selection_of_reco_in_path(
    path_to_reco='/path/to/reconstruction/',
    particle="H1",
    scan="S500",
    per_LLKf=0.2,
    per_stdrho=0.15,
    per_nbpixel=0.9,
    save_animation=True,
    create_mode=True,
    selection_method='looped'
)
```

---

## Suggestions Implemented
- Removed redundant imports for clarity.
- Improved function naming consistency.
- Enhanced directory and file handling using Python-native methods.
- Consolidated error handling and optimized logging.
- Improved plotting reliability and visualization clarity.

---

## Future Enhancements
- Implement further automation and efficiency via parallel processing.
- Add robust input validation.
- Expand metrics for selection beyond LLKf and stdrho.

---

## Testing

Execute tests with pytest:
```bash
pytest tests/test_selection_utilities.py
```

---

## License

Distributed under the MIT License. See [LICENSE](../../LICENSE) for more details.

