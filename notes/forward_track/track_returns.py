"""İleriye dönük getiri takibi — derin-değer adaylarının performansını ölç.

Amaç: Her tarama gününde `YYYY-MM-DD_adaylar_snapshot.csv` üretilir (aday listesi +
skorlar + entry_asof_date). Bu script, sonraki bir günde çalıştırıldığında her adayın
`entry_asof_date` kapanışından (giriş) `entry_asof` sonrası verinin son kapanışına (çıkış)
kadar getirisini hesaplar; varsa teknik hedef/stop tetiklenmesini işaretler.

Fiyat kaynağı (bu repo ortamında TradingView egress ile bloklu, borsapy doğrudan çalışmaz):
  - Colab/tvdatafeed ortamında:  --source tv   (otomatik çeker)
  - Bu ortamda: adayların OHLCV'sini BIST MCP get_historical_data ile çekip
    `notes/forward_track/prices/<SYMBOL>.tsv` (date,open,high,low,close,volume) olarak kaydet,
    sonra:  --source tsv --prices notes/forward_track/prices

Kullanım:
  python track_returns.py --snapshot notes/forward_track/2026-07-05_adaylar_snapshot.csv \
                          --source tsv --prices notes/forward_track/prices
  python track_returns.py --snapshot .../snapshot.csv --source tv   # Colab

Çıktı: aday başına satır (giriş, güncel, getiri_%, gün, hedef_orta_gördü, hedef_tam_gördü,
stop_gördü, max_lehte_%, max_aleyhte_%) + havuz/tip bazında ortalama getiri özeti.
"""
import argparse, os, glob
import pandas as pd
import numpy as np


def _load_tsv(prices_dir, sym):
    fp = os.path.join(prices_dir, f"{sym}.tsv")
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp, sep="\t", parse_dates=["date"]).set_index("date").sort_index()
    return df


def _load_tv(sym, n_bars=400):
    from tvDatafeed import TvDatafeed, Interval  # Colab
    tv = TvDatafeed()
    df = tv.get_hist(sym, "BIST", Interval.in_daily, n_bars=n_bars)
    if df is None or df.empty:
        return None
    df = df.rename(columns=str.lower)
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]]


def track_one(row, df):
    """entry_asof kapanışından son bara kadar ileri getiri + hedef/stop kontrolü."""
    asof = pd.to_datetime(row["entry_asof_date"]).normalize()
    fwd = df[df.index >= asof]
    if len(fwd) < 2:
        return None
    entry = float(fwd["close"].iloc[0])
    last = float(fwd["close"].iloc[-1])
    path = fwd.iloc[1:]  # giriş barından sonrası
    ret = (last / entry - 1) * 100
    max_up = (path["high"].max() / entry - 1) * 100 if len(path) else np.nan
    max_dn = (path["low"].min() / entry - 1) * 100 if len(path) else np.nan
    def _hit(level, up):  # up=True: high>=level ; up=False: low<=level
        if pd.isna(level) or not len(path):
            return None
        return bool((path["high"] >= level).any()) if up else bool((path["low"] <= level).any())
    return dict(
        entry_close=round(entry, 4), guncel_close=round(last, 4),
        gun=len(path), getiri_pct=round(ret, 2),
        max_lehte_pct=round(max_up, 2) if pd.notna(max_up) else None,
        max_aleyhte_pct=round(max_dn, 2) if pd.notna(max_dn) else None,
        hedef_orta_gordu=_hit(row.get("hedef_orta"), True),
        hedef_tam_gordu=_hit(row.get("hedef_tam"), True),
        stop_gordu=_hit(row.get("hard_stop"), False),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--source", choices=["tsv", "tv"], default="tsv")
    ap.add_argument("--prices", default="notes/forward_track/prices")
    args = ap.parse_args()

    snap = pd.read_csv(args.snapshot)
    rows, missing = [], []
    for _, r in snap.iterrows():
        sym = r["symbol"]
        df = _load_tv(sym) if args.source == "tv" else _load_tsv(args.prices, sym)
        if df is None:
            missing.append(sym); continue
        res = track_one(r, df)
        if res is None:
            missing.append(sym); continue
        rows.append({**{k: r[k] for k in ("symbol", "sector", "tip", "final_score",
                     "tech_ucuzluk", "banker", "temel_skor")}, **res})

    if not rows:
        print("Hiç fiyat verisi bulunamadı. --prices dizinine <SYMBOL>.tsv koy ya da --source tv kullan.")
        if missing: print("Eksik:", ", ".join(missing))
        return
    out = pd.DataFrame(rows).sort_values("getiri_pct", ascending=False)
    asof = snap["entry_asof_date"].iloc[0]
    print(f"=== İLERİ GETİRİ (giriş {asof} → güncel) | {len(out)} aday ölçüldü ===")
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
    print(out[["symbol", "tip", "final_score", "entry_close", "guncel_close", "gun",
               "getiri_pct", "max_lehte_pct", "max_aleyhte_pct",
               "hedef_orta_gordu", "stop_gordu"]].to_string(index=False))
    print(f"\nHAVUZ ort. getiri: {out.getiri_pct.mean():+.2f}%  medyan {out.getiri_pct.median():+.2f}%  "
          f"kazanan {100*(out.getiri_pct>0).mean():.0f}%  (n={len(out)})")
    print("Tip bazında:")
    for tip, g in out.groupby("tip"):
        print(f"  {tip:14} ort {g.getiri_pct.mean():+6.2f}%  medyan {g.getiri_pct.median():+6.2f}%  n={len(g)}")
    print("final_score >=60 alt kümesi:")
    hi = out[out.final_score >= 60]
    if len(hi):
        print(f"  ort {hi.getiri_pct.mean():+.2f}%  kazanan {100*(hi.getiri_pct>0).mean():.0f}%  n={len(hi)}")
    if missing:
        print("\nFiyat verisi bulunamayan:", ", ".join(missing))


if __name__ == "__main__":
    main()
