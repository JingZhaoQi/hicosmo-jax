# DES-SN5YR (DESY5) Supernova Data

## Data Source

Official data release: https://github.com/des-science/DES-SN5YR

**Reference**: DES Collaboration, 2024, ApJL, 973, L14 (arXiv: 2401.02929)
**Calibration**: Dovekie re-calibration (Popovic et al. 2026), supersedes original Vincenzi+2024

## Sample

- **Total SNe**: 1829 (1820 after quality cuts)
- **DES SNe**: 1635 photometrically classified
- **External low-z**: 194 from CfA, CSP, FOUND surveys
- **Redshift range**: 0.025 < z < 1.14
- **Standardization**: BBC + SALT3 (Tripp formula) with DOVEKIE calibration

## Files

### DES-Dovekie_HD.csv
SNANA-format data file with 1820 SNe (9 comment lines + VARNAMES + 1820 data lines).

**Key columns**:
- `zHD`: Hubble diagram redshift (CMB frame with peculiar velocity correction)
- `zHEL`: Heliocentric redshift
- `MU`: Distance modulus (corrected for bias and contamination, computed with H0=70)
- `MUERR`: Statistical uncertainty on MU
- `MUERR_SYS`: Diagonal systematic uncertainty (diagnostic)
- `PROBIA_BEAMS`: SNIa BEAMS probability from BBC (diagnostic)

### STAT+SYS.npz
Inverse covariance matrix (statistical + systematic) in compressed NPZ format.
- `nsn`: Number of SNe (1820)
- `cov`: Upper triangular elements of 1820x1820 inverse covariance matrix (float32)
- Format: `inv_cov[np.triu_indices(n)] = data['cov']`, then symmetrize

### STATONLY.npz
Inverse covariance matrix (statistical only) in same format as STAT+SYS.npz.

## Key Differences from Pantheon+

1. Data provides distance moduli (MU) directly, not apparent magnitudes (m_b_corr)
2. Covariance stored as **inverse** covariance in upper-triangular NPZ format
3. No SH0ES calibration mode
4. SNANA-format CSV (lines start with `SN:`) instead of whitespace-delimited

## Official Constraints

- **Flat LCDM (Dovekie)**: Omega_m = 0.330 +/- 0.015 (SN only, Dovekie re-calibration)
- **Flat LCDM (original)**: Omega_m = 0.352 +/- 0.017 (SN only, Vincenzi+2024, superseded)

## Citation

DES Collaboration, Abbott, T. M. C., et al. 2024, ApJL, 973, L14
"The Dark Energy Survey: Cosmology Results With ~1500 New High-redshift
Type Ia Supernovae Using The Full 5-year Dataset"

## Download Date

Data downloaded: 2026-03-28
