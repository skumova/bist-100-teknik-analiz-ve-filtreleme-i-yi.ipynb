# Derin-Değer Kontrarian Tarama — Çalışma Hafızası (2026-07-05)

> Yarın değerlendirmek üzere kaydedildi. Karar günlüğü + backtest bulguları + kod durumu.

## 1. Bugün ne yaptık (özet)
- **Banker (akıllı para) skorunu yeniden tasarladık** ve **çekirdekten çıkardık.**
- Point-in-time (look-ahead'siz) **backtest ile doğruladık**; önce 3, sonra 5, sonra 8 isimle (3'ü BIST-100 dışı) genişleterek koştuk.
- Mimariyi kod + notebook markdown'larına işledik, `claude/bist-technical-analysis-ol92sk` dalına push ettik.
- **Bugün yeni kod değişikliği yok** (son genişletilmiş backtest yalnızca doğrulama amaçlıydı; tasarımı değiştirmedi).

## 2. Nihai mimari (KARAR — backtest ile sabitlendi)
```
final = TEKNİK × zamanlama_teyidi × temel_katkı × trap_çarpanı
  çekirdek        = YALNIZCA teknik ucuzluk (sıralamayı bu belirler)
  zamanlama_teyidi = banker_çarpan(±%15) × RSI_diverjans_çarpan(+%8)
  temel_katkı     = 1 + 0.20×(temel−50)/50   (±%20)
  trap_çarpanı    = value-trap cezası (0.3–0.8× / diskalifiye)
```
- **Neden banker çekirdek DEĞİL:** eski seviye-tabanlı banker ucuzlukla ters çalışıyordu (`corr(teknik,banker) ≈ −0.5`); %40 çekirdeğe katmak öngörüyü ~yarıya düşürüyordu.
- **Yeni banker = dönüş/birikim odaklı** (seviye değil): A-D/OBV eğimi, CMF/MFI'nin 5-bar DÖNÜŞÜ, pozitif diverjans.
- **Parametreler** (`src/deep_value.py`): `TECH_GATE=60`, `BANKER_BONUS_STRENGTH=0.15`, `RSI_DIV_BONUS=0.08`, `FUND_BONUS_STRENGTH=0.20`.

## 3. Backtest bulguları (forward 40 gün, look-ahead yok)

### 3 isim (ULKER, GARAN, EREGL) — 207 gözlem
- corr(YENİ final, fwd) **+0.322** ≥ sade teknik +0.317 > eski 60/40 +0.229
- birikim bayrağı avantajı: **+5.5%**

### 5 isim (+ASELS savunma, +THYAO havacılık) — 345 gözlem
- YENİ +0.149 ≥ teknik +0.145 > 60/40 +0.113
- birikim avantajı: **+6.66%** (acc=True +11.0% vs +4.4%)

### 8 isim (+KARSN, +IZMDC, +EGEEN — 3'ü BIST-100 DIŞI) — 552 gözlem
- YENİ **+0.072** ≈ teknik +0.073 >> eski 60/40 **+0.032** (60/40 sinyali yarıya düşürüyor — tekrar teyit)
- `corr(teknik,banker) = −0.319` (hâlâ negatif → çekirdeğe koyma kararı doğru)
- birikim bayrağı avantajı (havuz): **+4.92%** (acc=True +8.34% vs +3.42%, n=68)
- RSI diverjansı avantajı: **+3.41%**
- teknik kovalar monotonik: <40 → +3.4% (%50 kaz.), 40-55 → +6.0% (%68), 55-70 → +9.4% (%80)
- **hisse-bazlı: yeni ≥ teknik veya çok yakın — 8/8 isim.** (corr'lar 0.21–0.67)

### İki dürüst nüans (yarın akılda tut)
1. **tech≥70 kovası +2.6%'ya düşüyor** (n=13): en dövülmüş küçük-caplar "düşen bıçak" olabiliyor → value-trap cezası + kademeli Fibonacci girişi ŞART.
2. **Zaten-ucuz (tech≥50) alt kümesinde birikim bu pencerede negatif** çıktı (−0.78% vs +5.30%, ama n=11 — gürültülü). Genel +4.92% ile çelişiyor → banker'ı **küçük (±%15) tutmanın** neden doğru olduğunu destekliyor.

## 4. Neden pooled corr düşük görünüyor (yanılma)
Farklı fiyat-ölçekli isimleri birlikte havuzlamak mutlak korelasyonu seyreltir (+0.32 → +0.07). Karar için önemli olan **göreli sıralama** (yeni ≥ teknik > 60/40) ve **hisse-bazlı korelasyonlar** (0.21–0.67); ikisi de her isimde tutuyor.

## 5. Kod / dosya durumu
- `src/deep_value.py` — motor güncel (yeni banker + composite). Rapor kolonu: `zamanlama_carpan` eklendi.
- `BIST_Deep_Value_Contrarian_Tarama.ipynb` — tamamen inline, yeni motor + güncel mimari markdown'ları. Colab'a tek dosya yüklenip çalışır.
- Notebook üretici: `scratchpad/build_nb2.py` (motoru `src/deep_value.py`'den inline eder; markdown'ı notebook'un kendisinden okur — md_by).
- Backtest scriptleri: `scratchpad/backtest_multi.py`, `scratchpad/backtest_final.py`.
- Backtest TSV'leri (scratchpad, **geçici konteyner** — kalıcı değil): ASELS, EGEEN, EREGL, GARAN, IZMDC, KARSN, THYAO, ULKER (1y, split-adjusted, borsapy/MCP).
- **Branch:** `claude/bist-technical-analysis-ol92sk` (son commit `f164b3f` — banker redesign). Bugünkü backtest için yeni commit YOK.

## 6. Ortam notu (yarın için önemli)
- Bu ortamda **TradingView (data.tradingview.com) egress politikası ile 403** — borsapy'nin doğrudan Python history çağrısı çalışmıyor. **Veri yalnızca BIST MCP tool'u (`get_historical_data`) üzerinden** geliyor (server-side). Bu yüzden geniş universe backtest'i pahalı (her isim context'e dökülüyor).
- Notebook Colab'da tvdatafeed ile çalışıyor (orası TV'ye erişebiliyor); backtest bu ortamda MCP ile yapıldı.

## 6b. Bugünkü CANLI tarama çıktısı (2026-07-05, entry_asof 2026-07-03)
- Taranan **602** hisse · aşırı-ucuz kapıyı geçen **109** · aday **25** · tuzak **129** · diskalifiye 0.
- Aday tipleri: birikim 13, birikim-spek 7, birikim+dip 5.
- **İleri getiri takibi KODA push EDİLMİYOR (kullanıcı tercihi).** Takip sohbet üzerinden yürür:
  kullanıcı her tarama günü çıktıyı (Excel/aday listesi) paylaşır; kohort olarak kaydedilir,
  ≥2 gün birikince adayların giriş→güncel getirisi (MCP fiyatlarıyla) hesaplanıp skor tablosu tutulur.
  Kohort mantığı: her tarama günü = ayrı kohort, kendi giriş gününden T+3/5/10/20'de ölçülür;
  günler eklendikçe hem yeni kohort eklenir hem eski kohortlar olgunlaşır.
- **En yüksek final skorlu adaylar:** GENTS 93.7 (Sanayi, birikim+dip, tech 79/banker 63/temel 84),
  FORMT 81.9 (Madencilik), LMKDC 77.8 (Madencilik, birikim+dip), KRGYO 73.6 (GYO),
  IHGZT 71.7, KZBGY 71.3, IHLGM 70.2 (temel 95, güv.marjı %168), DERHL 65.8 (birikim-spek).
- **Motor doğru çalışıyor teyidi:** en dövülmüş RSI'lar (KONTR 25.6, TRILC 25.1, CANTE 27.4,
  ALGYO 24.9, OBAMS 24.7) aday DEĞİL → hepsi tuzak (UCUZ_CUNKU_ZARAR / EV/EBITDA_PAHALI /
  DUSEN_BICAK). Yani "en ucuz ≠ en iyi" mantığı sahada işliyor.
- Not: `Alim_Plani` sayfası yalnızca 25 üst-skorlu isme merdiven kuruyor; 8'i adaylarla örtüşüyor.
  Kalan 17 adayın giriş fiyatı takip anında geçmişten okunacak (tracker bunu yapıyor).

## 7. Yarın için açık başlıklar (öneri)
- [ ] **İlk forward check (sohbette):** kullanıcı yeni gün çıktısını paylaşınca 2026-07-05 kohortunun
      giriş→güncel getirisini hesapla; tip (birikim/spek/dip) ve final_score kovalarına göre ayrışma var mı bak.
- [ ] Karar: mevcut tasarımı dondurup canlı tarama çıktısını mı değerlendirelim, yoksa `BANKER_BONUS_STRENGTH`/`RSI_DIV_BONUS` katsayılarını daha geniş universe ile ince mi ayarlayalım?
- [ ] tech≥70 "düşen bıçak" kovası için ek filtre (ör. dip_confirm zorunlu) test edilebilir.
- [ ] İstenirse likidite eşiği (`MIN_LIQ`) ve sektör çeşitlendirme (`MAX_SEKTOR`) knob'larını canlı çıktı üstünde kalibre et.
- [ ] Value-trap eşiklerini gerçek çıktı üstünde gözden geçir.
