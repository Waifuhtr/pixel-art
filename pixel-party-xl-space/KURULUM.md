# Kurulum ve Dağıtım Rehberi

## 1. Space'i oluştur

huggingface.co → **New Space**

| Alan | Değer |
|---|---|
| SDK | **Docker** → *Blank* şablonu |
| Hardware | Şimdilik **CPU basic** (ücretsiz) |
| Visibility | Tercihine göre |

> Hardware'ı **şimdi T4 yapma.** Sebebi 3. adımda.

## 2. Dosyaları yükle

Bu klasörün **içindekileri** Space repo'sunun köküne koy — `pixel-party-xl-space/`
klasörünü değil, içindeki dosyaları:

```
README.md
Dockerfile
requirements.txt
download_models.py
pipeline.py
app.py
.dockerignore
static/index.html
static/style.css
static/app.js
```

Web arayüzünden yüklüyorsan `static/` klasörünü ayrıca oluşturman gerekmez —
dosya adını `static/index.html` olarak yazman yeterli, HF klasörü kendisi açar.

Git ile:

```bash
git clone https://huggingface.co/spaces/KULLANICI_ADIN/SPACE_ADIN
cd SPACE_ADIN
# bu klasörün içeriğini buraya kopyala
git add -A && git commit -m "Pixel Party XL space" && git push
```

**Önemli:** `README.md`'nin en üstündeki `---` ile çevrili YAML bloğu Space'in
yapılandırması. Silme veya değiştirme — özellikle `sdk: docker` ve
`app_port: 7860` satırlarını.

## 3. Donanımı seç — kredi açısından kritik kısım

Build, Space'e atanmış donanımda çalışır. Yani **T4 seçili haldeyken build
alırsan, build süresi boyunca da GPU ücreti işler.**

Bu Space'in build'i uzun tarafta: torch + CUDA (~3 GB) ve model ağırlıkları
(~7 GB) indiriliyor. Toplam **~15-25 dakika**.

Doğru sıra:

1. **CPU basic** üzerindeyken dosyaları push et, build'in bitmesini bekle
2. Loglarda `[download] all weights cached` satırını gör
3. **Sonra** Settings → Hardware → **Nvidia T4 small**'a geç

Donanım değişince container yeniden başlar ama **image yeniden build edilmez** —
ağırlıklar image'ın içinde olduğu için sadece ~40 saniyelik bir açılış yaşarsın.
Bu şekilde 20 dakikalık build'i ücretsiz donanımda geçirmiş olursun.

> CPU basic'te build sırasında "app çalışmıyor" uyarısı görürsen sorun değil —
> model CPU'da da yüklenir, sadece üretim çok yavaştır. Amaç sadece image'ı
> hazırlamak.

## 4. Uyku ayarı

Settings → **Sleep time** → 15 dakika (veya daha kısa).

GPU faturası Space uyanıkken işler. Uyuduktan sonra ilk istekte ~40 saniyelik
bir açılış olur; ağırlıklar image'da olduğu için tekrar indirme yapılmaz.

## 5. Beklenen performans (T4 small)

| İş | Süre |
|---|---|
| Soğuk açılış (uykudan uyanma) | ~40 sn |
| 1024×1024, 25 adım, 1 görsel | ~18-25 sn |
| 1024×1024, 25 adım, 4 görsel | ~75-100 sn |
| 1344×768, 25 adım, 1 görsel | ~20-28 sn |

VRAM kullanımı ~7 GB (ağırlıklar) + ~3 GB (aktivasyonlar) ≈ 10 GB. 16 GB'lık
T4'te rahat bir pay bırakıyor; üretim her zaman tek görsellik partiler halinde
yapıldığı için görsel sayısını artırmak VRAM'i yükseltmez, sadece süreyi uzatır.

## 6. Kullanım ipuçları (model kartından)

- Prompt İngilizce olmalı. `. in pixel art style` eki otomatik ekleniyor.
- Negatif prompt varsayılanı `mixels. amateur. multiple` — model kartının
  önerdiği değer, değiştirmeden bırakman genelde daha iyi.
- **Küçültme'yi 8× bırak.** Model 1024px'te çiziyor ama gerçek çıktı 128px'lik
  sprite. 8× küçültme olmadan elinde sadece bulanık bir büyütme kalır.
- Küçük ikon/eşya için 8×, büyük sahne için 4× dene.
- Palet sınırlaması (16-32 renk) sprite'ları daha "otantik" gösterir.
- Bir sonucu beğendiysen **INIT →** ile başlangıç görseli yap ve düşük değişim
  gücüyle (0.3-0.5) varyasyon üret. Model kartı init görsellerinin bu modelde
  çok işe yaradığını söylüyor.

## 7. Sorun giderme

| Belirti | Sebep / çözüm |
|---|---|
| Build'de `no space left on device` | Space'in disk kotası dolmuş; Factory rebuild dene |
| Açılışta `CUDA out of memory` | Donanım T4 değil, daha küçük GPU seçilmiş |
| Görseller tamamen siyah | fp16 VAE devre dışı kalmış — `pipeline.py`'de VAE repo'sunun `madebyollin/sdxl-vae-fp16-fix` olduğunu doğrula |
| "Model yükleniyor" takılı kalıyor | Logları aç; `[engine] load failed:` satırı gerçek hatayı verir |
| `429 Queue is full` | Aynı anda 12'den fazla iş var; normal koruma |
| Arayüz açılıyor ama üretim 503 | Model henüz yüklenmemiş, ~40 sn bekle |

## 8. Ayarlanabilir limitler

`app.py` başındaki sabitler:

```python
MAX_IMAGES = 4              # tek istekte üretilecek görsel
MAX_QUEUE = 12              # kuyruk derinliği
MAX_ACTIVE_PER_CLIENT = 2   # IP başına eşzamanlı iş
RESULT_BYTE_BUDGET = 384MB  # RAM'de tutulan sonuç bütçesi
MAX_JOBS_KEPT = 40          # saklanan iş sayısı
```

Space'i herkese açık yapacaksan bu değerleri düşürmek GPU harcamanı sınırlar.
