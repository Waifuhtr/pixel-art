---
title: Pixel Art Studio
emoji: 🎨
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Pixel Art Studio — yerel/ücretsiz Texel Studio motoru

Bu depo, [Texel Studio](https://github.com/EYamanS/texel-studio)'nun açık kaynak
motorunun bir **fork**'udur. Tek fark: Gemini API'sine bağımlı olan iki adım
yerine, tamamen yerel/ücretsiz çalışan iki model kullanır — bir Hugging Face
Space container'ı içinde, API anahtarı gerekmeden.

> Bu repo **arayüzü içermez.** Arayüz (HTML/CSS/JS) ayrı bir zip olarak
> teslim edildi — kurulum için aşağıdaki "Hugging Face Space'e kurulum"
> bölümüne bakın.

## Mimari — iki farklı model, iki farklı iş

Orijinal projenin tanıtım videosundaki "AI pikselleri tek tek çizip geri
adım atıp kontrol ediyor" animasyonu bir görsel üretim (diffusion) modelinden
**gelmiyor** — bir LLM'in araç çağırma (tool-calling) döngüsünden geliyor
(`agent.py`, LangGraph `create_react_agent`). Bir diffusion modeli (SD-Turbo
gibi) piksel piksel çizim kararı veremez; tüm görseli 1-4 adımda üretir.
Bu yüzden burada iki ayrı, birbirinin yerini tutmayan model çalışıyor:

| Adım | Model | Neden |
|---|---|---|
| Concept art / referans görsel (tek seferlik) | **SD-Turbo GGUF (Q8_0)**, CPU'da `stable-diffusion.cpp` ile | Artık Gemini'ye ihtiyaç yok |
| Piksel piksel boyama (canlı, `draw_pixel`/`fill_rect`/... araç çağrıları) | **Ollama** ile yerel bir LLM (`OLLAMA_MODELS`) | Bu adım zaten orijinal kodda Ollama'yı destekliyordu — ek kod gerekmedi |

İkisi de aynı container'da, aynı 8 vCPU / 32GB donanımda, GPU olmadan
çalışır. Bu depoda yaptığım tek gerçek kod değişikliği, concept art adımını
(`server.py` `/api/reference`, `worker.py` `handle_reference`,
`jobs/sprite_reference.py`) `local_sd.py` üzerinden SD-Turbo'ya yönlendirmek —
bkz. `USE_LOCAL_SD` ortam değişkeni. Ajan tarafı (`agent.py`) zaten
değişmeden çalışıyor.

## Performans beklentisi (önemli, dürüst olmak gerekirse)

GPU yok — her şey CPU'da çalışıyor:
- **SD-Turbo Q8 (2.3GB), 512x512, 4 adım:** CPU'da görsel başına
  büyük olasılıkla **saniyeler-onlarca saniye** sürer (tek seferlik, hızlı).
- **Ollama ajanı:** Bir sprite genelde 10-80 araç-çağrı adımı gerektiriyor
  (`agent.py`'deki `max_steps=80`). CPU'da 4-8B parametreli bir modelle bu,
  sprite başına **birkaç dakikadan on dakikaya** kadar sürebilir. Bu bir hata
  değil — CPU'da LLM çıkarımının doğal maliyeti. Daha küçük bir model
  (`.env`'de varsayılan) daha hızlı ama daha "aptal" sonuç verir;
  `qwen3:8b`/`llama3.1:8b` daha iyi ama daha yavaş.
- İlk konteyner açılışı ayrıca SD-Turbo GGUF dosyasını (~2.3GB) ve Ollama
  modelini indirir — ilk istek birkaç dakika gecikebilir, sonraki
  çalıştırmalarda diskte kalırsa (bkz. "Kalıcı depolama" altında) tekrar
  inmez.

## Bu depo build+deploy edilmeden önce doğrulanmadı

Bu kodu Texel Studio'nun gerçek API sözleşmesine (SSE event formatı, iş
akışı) sadık kalarak ve `stable-diffusion-cpp-python` ile Ollama'nın
dokümante edilmiş API'lerine göre yazdım, ama bu sandbox içinde GPU'suz bir
HF Space'i simüle edip uçtan uca build alamadım (çok büyük indirme +
build süresi + gerçek bir Space'e ihtiyaç var). İlk deploy'da küçük bir
Docker/paket sürümü sorunu çıkarsa şaşırmayın — HF Space'in "Logs" sekmesi
sorunu hemen gösterir, söylerseniz birlikte düzeltiriz.

## Hugging Face Space'e kurulum

**Önemli:** HF Space'in Docker build'i yalnızca **Space'in kendi git
deposundaki** dosyalara bakar — bu GitHub deposundan haberi yoktur. Yani
Space'e sadece arayüz zip'ini atarsanız `Dockerfile`, `server.py`,
`local_sd.py`, `entrypoint.sh` (modeli indiren kod) ortada olmaz; build
başarısız olur ya da HF Docker SDK algılayamayıp siteyi çıplak statik HTML
olarak sunar — bu durumda `app.js`'in çağırdığı `/api/...` uçları 404
döner. Backend'i (bu depo) ve arayüzü (ayrı verilen zip) **birlikte**,
aynı Space deposuna koymanız gerekiyor. İki yol:

### A) Git ile (önerilen — sonraki güncellemeler için de kolay)

```bash
git clone --branch claude/texel-studio-analysis-9u37oy https://github.com/Waifuhtr/pixel-art.git
cd pixel-art

# Arayüz zip'ini açıp static/ içine koyun (placeholder index.html'in üzerine)
unzip ~/Downloads/pixel-art-studio-frontend.zip -d /tmp/frontend
cp /tmp/frontend/index.html /tmp/frontend/style.css /tmp/frontend/app.js static/

# huggingface.co'da SDK: Docker seçerek BOŞ bir Space oluşturun, adresini not edin,
# sonra o Space'i ikinci bir git remote olarak ekleyip push edin:
git remote add space https://huggingface.co/spaces/<kullanici-adiniz>/<space-adi>
git push space claude/texel-studio-analysis-9u37oy:main
```

Push için bir Hugging Face access token gerekir (huggingface.co/settings/tokens
→ "Write" yetkili token; `huggingface-cli login` ile bir kere girip
credential helper'a bıraktırabilir, ya da URL'ye
`https://<kullanici>:<token>@huggingface.co/spaces/...` şeklinde gömebilirsiniz —
token'ı hiçbir yere commit etmeyin).

### B) Tarayıcıdan sürükle-bırak (git kullanmadan)

1. GitHub'da bu depo → **Code → Download ZIP**, indirip açın.
2. Ayrı verdiğim arayüz zip'ini açın, içindeki `index.html`, `style.css`,
   `app.js` dosyalarını, az önce açtığınız klasörün **`static/`**
   alt klasörüne kopyalayın (placeholder `index.html`'in üzerine).
3. huggingface.co'da SDK: **Docker** ile yeni bir Space oluşturun.
4. Space sayfası → **Files** sekmesi → **Add file → Upload files** →
   açtığınız klasörün TÜM içeriğini (Dockerfile, `*.py` dosyaları,
   `jobs/` klasörü, içi dolu `static/` klasörü, `requirements.txt`,
   `README.md`, vb.) tek seferde sürükleyip bırakın → commit edin.

Her iki yolda da commit atılır atılmaz HF otomatik Docker build'i
başlatır; `entrypoint.sh` container ayağa kalkınca SD-Turbo GGUF'u ve
Ollama modelini kendi indirir (ilk açılış birkaç dakika sürer).

Ardından Space "Settings → Variables and secrets" kısmından
`.env.example`'daki değişkenleri girin (hepsi opsiyonel varsayılanlarla
gelir, hiçbirini girmeseniz de local-SD + local-Ollama ile ayağa kalkar).

### Kalıcı depolama (opsiyonel)

HF Space'in ücretsiz/temel diskleri container her yeniden başladığında
sıfırlanabilir — bu durumda SD-Turbo GGUF'u ve Ollama modelini her
restart'ta yeniden indirir (birkaç dakika ek gecikme, hata değil). Eğer
Space'inizde "Persistent Storage" açıksa (`/data` bağlanır),
`SD_MODEL_DIR=/data/sd-models` ayarlayıp Ollama'yı da
`OLLAMA_HOME=/data/ollama` gibi bir dizine yönlendirerek indirmeleri
kalıcı hale getirebilirsiniz.

## Attribution / Lisans

Bu depo [texel-studio](https://github.com/EYamanS/texel-studio)'nun
kaynak kodunun büyük kısmını (agent.py, server.py, jobs/, worker.py,
storage.py) neredeyse değiştirmeden içeriyor; orijinal `LICENSE` dosyası
korunmuştur. O lisansa göre: kendi kullanımınız için self-host etmek,
değiştirmek serbest; bunu texel.studio ile **rekabet eden ücretli bir
SaaS** olarak sunmak yasak; yeniden dağıtırken orijinal depoya link
vermek gerekiyor (bu README bunu yapıyor). Orijinal proje:
**https://github.com/EYamanS/texel-studio**

## Ortam değişkenleri

`.env.example` dosyasına bakın — yerel/ücretsiz varsayılanlar zaten
açık (`USE_LOCAL_SD=true`, Ollama). Gemini/OpenAI'ye geri dönmek
isterseniz ilgili API anahtarını girip `USE_LOCAL_SD`'yi kaldırmanız
yeterli.
