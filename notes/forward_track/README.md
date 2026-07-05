# İleriye Dönük Getiri Takibi (forward tracking)

Amaç: her tarama gününün adaylarını sabitleyip **sonraki günlerde gerçek getirisini** ölçmek —
motorun kararlarını canlı, look-ahead'siz doğrulamak.

## Akış
1. **Tarama günü:** Notebook çıktısındaki `Adaylar` (+`Alim_Plani`) sayfalarından bir snapshot
   üretilir: `YYYY-MM-DD_adaylar_snapshot.csv` (aday listesi, skorlar, `entry_asof_date`,
   ve varsa teknik hedef/stop). Fiyat SAKLANMAZ — giriş fiyatı kontrol gününde geçmişten okunur.
2. **Kontrol günü:** `track_returns.py` her adayın OHLCV'sini alır, `entry_asof_date` kapanışını
   giriş, son barı çıkış kabul edip getiriyi + hedef/stop tetiklenmesini hesaplar.

## Fiyatı nereden alır
- **Colab / tvdatafeed:** `python track_returns.py --snapshot <csv> --source tv` (otomatik çeker).
- **Bu ortam (TradingView egress bloklu):** adayların OHLCV'sini BIST MCP `get_historical_data`
  ile çekip `prices/<SYMBOL>.tsv` (date,open,high,low,close,volume) olarak kaydet, sonra:
  `python track_returns.py --snapshot <csv> --source tsv --prices notes/forward_track/prices`

## Snapshot'lar
- `2026-07-05_adaylar_snapshot.csv` — entry_asof 2026-07-03, 25 aday (aşağıya bak).

Yeni tarama günlerinde aynı formatta yeni snapshot ekle; tracker hepsini ayrı ayrı ölçebilir.
