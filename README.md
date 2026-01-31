# Implied-Vol-Surface

Tools to build and visualize implied-volatility smiles and surfaces from option chain CSVs.

## Programs

### `xget_option_prices.py`
Download option chain data and save to CSV for use by the other tools. It is taken from my
[Yahoo-Option-Prices](https://github.com/vivek-v-rao/Yahoo-Option-Prices) project and 
documented more extensively there.

Key features:
- Fetches option chains (per expiry or expiry range).
- Saves a combined CSV with calls and puts and summary stats.
- Optional plot of implied vol vs strike.

Usage:
```
python xget_option_prices.py SYMBOL [--expiry YYYYMMDD|YYYYMMDD:YYYYMMDD] [--exp-range N:M]
    [--plot [file.png]] [--summary] [outfile.csv]
```

Example:
`python xget_option_prices.py "^SPX"`

downloads the option chain for SPX to `SPX_all.csv` and gives [summary data](results/SPX_option_summary.txt).

### `ximplied_vols.py`
Compute implied vols by expiry from option chain data (CSV or freshly downloaded from Yahoo Finance).

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

Example:
`python ximplied_vols.py spx_all.csv --expiry 20260206 20260213 20260220 --fwd-range 0.9:1.1 --plot`
creates [raw implied vol plots](plots/spx_iv.png) and [tables](results/spx_implied_vols.txt).

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
Example:
`python xsvi.py spx_all.csv --expiry 20260202 20260206 20260213 20260220 --fwd-range 0.9:1.1 --plot`
creates [SVI implied vol curves](plots/spx_svi.png).

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
Example:
`python xssvi.py spx_all.csv --expiry 20260202 20260206 20260213 20260220 --fwd-range 0.9:1.1 --plot`
creates [SSVI implied vol curves](plots/spx_ssvi.png).
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
Example:
`python xsabr.py spx_all.csv --expiry 20260202 20260206 20260213 20260220 --fwd-range 0.9:1.1 --plot`
creates [SABR implied vol curves](plots/spx_sabr.png).

### `xcev.py`
Fit CEV smiles per expiry from European option data.

Key features:
- CEV prices via noncentral chi-square formula.
- Fits sigma and beta by default (use `--fixed-beta` to hold beta fixed).
- Plots fitted IV curves and implied densities.
- Infers symbol from `contractSymbol` in the input file for plot titles (when available).

Usage:
```
python xcev.py options.csv [--expiry YYYY-MM-DD|YYYYMMDD:YYYYMMDD] [--as-of YYYY-MM-DD]
    [--fwd-range L:U] [--both-sides] [--beta B] [--fixed-beta]
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
