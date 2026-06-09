# LEAP Notebooks

Interactive Jupyter notebooks for generating magnetic field predictions along Europa Clipper flyby trajectories.

## Quickstart

**`quickstart_clipper.ipynb`** - Predict magnetic fields for Europa Clipper flybys with customizable plasma conditions.

### Setup

1. **Clone and install:**
   ```bash
   git clone https://github.com/reddy-sachin/LEAP.git
   cd LEAP
   conda env create -f environment.yml
   conda activate leap_env
   ```

2. **Launch notebook:**
   ```bash
   jupyter notebook notebooks/quickstart_clipper.ipynb
   ```

3. **Run cells sequentially**
   - Model weights download from HuggingFace (`reddysachin/LEAP`)
   - Flyby trajectories download from HuggingFace (`reddysachin/Europa_Clipper_Tour`)
   - First run downloads ~50MB, subsequent runs use cache

### Usage

1. Select a flyby from the table (48 Europa Clipper flybys available)
2. Set plasma parameters in the input cell:
   - **RhoO**: Magnetospheric O+ mass density (320-2400 amu/cm³)
   - **H0**: Atmospheric scale height (2.5-7.5 × 10⁷ km)
   - **n0**: Atmospheric surface density (33-330 cm⁻³)
3. Toggle upstream field variability (±5% applied to Bx_U, By_U, Bz_U)
4. Run prediction cell to generate 5-panel plot

### Outputs

**5-panel stacked plot:**
- **R**: Radial distance (Europa radii)
- **Bx, By, Bz**: Magnetic field components (nT)
- **|B|**: Total field magnitude (nT)

Each panel shows:
- Background field (BJ) - gray dashed line
- Total field (Pred + BJ) - colored solid line

**Summary statistics** print to console:
- Closest approach altitude and time
- Field perturbations (δBx, δBy, δBz) at C/A
- Total field values and upstream conditions

**Export**: Uncomment final cell to save predictions to CSV

### Available Flybys

48 Europa Clipper flybys from the baseline tour (2031-2034):
- Altitudes: 23.9 km (closest) to 123.9 km
- ~480 time points per flyby (5-second cadence)
- Full trajectory, upstream, and background field data

### Troubleshooting

**Import errors?**
- Ensure you're in the LEAP directory: `cd LEAP`
- Activate environment: `conda activate leap_env`

**HuggingFace download fails?**
- Check internet connection
- Files cache to `~/.cache/huggingface/hub/`

**Plot not displaying?**
- Re-run the plot function cell (cell 10)
- Check matplotlib backend: `%matplotlib inline`

### Next Steps

- Run parameter surveys (template in cell 15)
- Compare predictions with magnetometer data
- Modify for JUICE or Galileo flybys (adapt CSV format)

For questions, see the main [README](../README.md) or open an issue on GitHub.
