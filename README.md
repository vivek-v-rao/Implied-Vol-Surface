# Implied-Vol-Surface

Tools to build and visualize implied-volatility smiles and surfaces from option chain CSVs.

## Programs

### `ximplied_vols.py`
Compute implied vols by expiry from option chain data (CSV or freshly downloaded).

Key features:
- Infer forward and discount factor using put-call parity.
- Compute implied vols for calls/puts and summarize ATM vols by tenor.
- Optional plotting of vol curves and ATM vol vs tenor.
- Supports American options with Bjerksund-Stensland (default) or CRR.

Usage (CSV):
```
python ximplied_vols.py options.csv [--expiry YYYY-MM-DD|YYYYMMDD:YYYYMMDD] [--as-of YYYY-MM-DD]
    [--fwd-range L:U] [--plot [file.png]] [--plot-atm [file.png]] [--both-sides]
    [--american] [--american-method bjerksund|crr] [--tree-steps N] [--spot S]
```

Usage (download + compute):
```
python ximplied_vols.py --symbol SYMBOL [--expiry YYYYMMDD|YYYYMMDD:YYYYMMDD] [--exp-range N:M]
    [--fwd-range L:U] [--plot [file.png]] [--plot-atm [file.png]] [--american]
```

### `xsvi.py`
Fit SVI smiles per expiry from European option data.

Key features:
- Fits SVI parameters per expiry and prints RMSE.
- Outputs fitted vols by strike (optional).
- Plots fitted IV curves and implied densities.
- Infers symbol from `contractSymbol` in the input file for plot titles (when available).

Usage:
```
python xsvi.py options.csv [--expiry YYYY-MM-DD|YYYYMMDD:YYYYMMDD] [--as-of YYYY-MM-DD]
    [--fwd-range L:U] [--both-sides] [--outfile out.csv]
    [--plot [file.png]] [--plot-density [file.png]] [--plot-density-log-s [file.png]]
```

### `xssvi.py`
Fit a surface SVI (SSVI) or eSSVI across expiries from European option data.

Key features:
- Global SSVI fit across tenors.
- Optional eSSVI with theta(T)=a+b*T^c and constant rho/eta/gamma.
- Plots fitted curves and implied densities (linear or log stock scale).
- Infers symbol from `contractSymbol` in the input file for plot titles (when available).

Usage:
```
python xssvi.py options.csv [--expiry YYYY-MM-DD|YYYYMMDD:YYYYMMDD] [--as-of YYYY-MM-DD]
    [--fwd-range L:U] [--both-sides] [--essvi]
    [--plot [file.png]] [--plot-density [file.png]] [--plot-density-log-s [file.png]]
    [--outfile out.csv]
```

### `xsabr.py`
Fit SABR smiles per expiry from European option data.

Key features:
- Hagan SABR implied-vol approximation.
- Beta defaults to 1.0 (lognormal); optional `--fit-beta`.
- Plots fitted IV curves and implied densities.
- Infers symbol from `contractSymbol` in the input file for plot titles (when available).

Usage:
```
python xsabr.py options.csv [--expiry YYYY-MM-DD|YYYYMMDD:YYYYMMDD] [--as-of YYYY-MM-DD]
    [--fwd-range L:U] [--both-sides] [--beta B] [--fit-beta]
    [--outfile out.csv] [--plot [file.png]]
    [--plot-density [file.png]] [--plot-density-log-s [file.png]]
```

### `xsummarize_options.py`
Print a per-expiry summary from an option chain CSV.

Usage:
```
python xsummarize_options.py options.csv
```


## Input CSV format

All programs expect a CSV with at least:
- `expiration` (YYYY-MM-DD)
- `strike`
- `option_type` (`call` or `put`)
- `bid`, `ask`

Optional:
- `lastPrice`
- `volume`, `openInterest`

## Notes

- Use quotes for symbols starting with `^` (e.g., `"^SPX"`).
- Density plots are computed via second derivatives of fitted Black-Scholes call prices; they can be sensitive to noisy fits.