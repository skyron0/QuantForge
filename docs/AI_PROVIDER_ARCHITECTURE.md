# QuantForge Yapay Zeka Sağlayıcısı ve Akıl Yürütme Altyapısı

Bu doküman, QuantForge platformunun Yapay Zeka Sağlayıcı Katmanı ve Akıl Yürütme Çalışma Zamanı (Sprint 2.9) mimarisini açıklamaktadır. Tasarım, sağlayıcı bağımsızlığı, güvenli kapanış (fail-closed) ve sıfır işlem yetkisi prensiplerine göre oluşturulmuştur.

---

## 1. Mimarî Katmanlar ve Sorumluluklar

Tüm yapay zeka işlemleri, alım-satım yetkisinden tamamen arındırılmış bağımsız bir altyapı üzerinde çalışır.

```
       +---------------------------------------------+
       |               ReasoningEngine               |
       |  (Prompts formatlama, Doğrulama Zinciri)    |
       +---------------------------------------------+
                              ↓
                  [ BaseAIProvider Arayüzü ]
                              ↓
                 +--------------------------+
                 |    AIProviderRegistry    |
                 +--------------------------+
                              ↓
              +-------------------------------+
              |         OllamaProvider        |
              | (requests, HTTP JSON, Format) |
              +-------------------------------+
```

### Katman Sorumlulukları

1. **ReasoningEngine**: `ReasoningRequest` girdilerini alır, sürüm kontrollü prompt şablonlarını kullanarak sorguyu formüle eder, seçili sağlayıcıyı çalıştırır, çok adımlı doğrulama zincirini çalıştırarak çıktıyı doğrular ve `ReasoningResult` döner.
2. **BaseAIProvider / Registry**: Sağlayıcı bağımsız çalışma zamanı kontratlarını ve sağlayıcıların isimle kaydolmasını sağlayan merkezi yapıyı sunar.
3. **OllamaProvider**: Ollama servisinin `/api/generate` uç noktasıyla HTTP üzerinden haberleşen, format kontrollerini ve hata dönüştürmelerini sağlayan somut adaptördür.

---

## 2. Çok Adımlı Doğrulama Zinciri (Validation Pipeline)

Yapay zekadan dönen ham metinler asla doğrudan güvenilerek karar mekanizmalarına iletilmez. ReasoningEngine aşağıdaki doğrulamaları arka arkaya koşturur:

```
[ Raw Provider Response ]
           ↓
    [ JSON Parsing ]        --> Hata durumunda StructuredOutputError fırlatılır. Led to retry.
           ↓
[ Response Schema Validation ] -> JSON Schema (jsonschema) yapısı doğrulanır.
           ↓
  [ Domain Validation ]      -> Alan değer tipleri ve güvenilirlik (confidence)
           ↓                    değerinin [0.0, 1.0] aralığında olduğu teyit edilir.
  [ ReasoningResult ]
```

### Yeniden Deneme (Bounded Retry) Mekanizması

Doğrulama zincirinde oluşan biçimsel (`StructuredOutputError`) hatalarında, yapay zekaya yapılandırılmış limit düzeyinde (`AI_STRUCTURED_MAX_RETRIES`, varsayılan: 3) yeniden deneme hakkı tanınır. Limit aşıldığında veya ağ hatası oluştuğunda sistem güvenli şekilde durur (fail-closed) ve işlem yapmaz.

---

## 3. Komut Şablonları (Prompt Versioning)

Sistemdeki yapay zeka komutları ve şablonları kodun farklı yerlerine serpiştirilmek yerine `backend/intelligence/reasoning/prompts.py` içinde sürüm kontrollü olarak tutulur. Her şablon şu alanları içerir:

- `prompt_id`: Şablon anahtarı (örn. `market_reasoning`).
- `version`: Sürüm etiketi (örn. `v1`).
- `task_type`: Görev tipi (`reasoning`).
- `system_template` / `user_template`: Yapay zekaya gönderilecek komutlar.
- `response_schema`: jsonschema standartlarında çıktı json yapısı.

Analiz sonuçlarının tekrarlanabilirliği ve audit işlemlerinde kullanılmak üzere, her başarılı `ReasoningResult` içerisinde kullanılan `prompt_id` ve `prompt_version` bilgileri muhafaza edilir.

---

## 4. Telemetri ve Hata Davranışı

### Telemetri Özellikleri (`AITelemetry`)

- Başarılı ve başarısız tüm yapay zeka isteklerini, gecikme sürelerini (latency_ms), hata tiplerini ve komut sürümlerini saklar.
- **Güvenlik Sınırı**: Kimlik bilgileri, API anahtarları veya hassas ham piyasa verileri telemetri loglarında yer almaz.

### Hata Kategorileri ve Davranışları

- **ProviderUnavailableError**: Ollama veya uzak yapay zeka sunucusu kapalı olduğunda tetiklenir.
- **ProviderTimeoutError**: Bağlantı veya çıkarım (inference) işleminde zaman aşımı oluştuğunda tetiklenir.
- **ProviderResponseError**: Yapay zekadan 200 harici geçersiz bir durum kodu dönüldüğünde fırlatılır.
- **ProviderConfigurationError**: `AI_MODEL` tanımının eksik veya geçersiz olduğu durumlarda çalışmayı durdurur.

---

## 5. Güvenlik Sınırı ve İşlem Yetkisi Yasağı

Yapay zeka katmanının platform üzerindeki rolü tamamen **gözlemci ve analitiktir**. Yapay zeka modülü:

- Doğrudan borsa veya broker kütüphanelerine erisemez.
- Alım-satım emirleri oluşturamaz veya portföy işlemlerini doğrudan yönlendiremez.
- `backend/intelligence` altındaki hiçbir sınıf, `backend/execution` veya `backend/broker` kütüphanelerini ithal edemez. Bu durum birim testleri içerisindeki statik ithalat analizi ile doğrulanmaktadır.

---

## 6. Yeni Sağlayıcı Ekleme Kılavuzu

Gelecekte yeni bir yapay zeka sağlayıcısı (örn. vLLM veya OpenAI) eklemek için aşağıdaki adımlar takip edilir:

1. **Concrete Provider Oluşturma**: `backend/intelligence/providers/` altında `BaseAIProvider` sınıfından türeyen yeni bir sınıf tanımlanır.
2. **Arayüz Implementasyonu**: Sınıfta `health_check()`, `generate()`, `generate_structured()` metotları ve `provider_name`, `model_name` özellikleri uygulanır.
3. **Kayıt**: Yeni sağlayıcı sınıfı `backend/intelligence/__init__.py` içinde `AIProviderRegistry.register("provider_adi", YeniProviderClass)` şeklinde kayıt edilir.
4. **Yapılandırma**: `.env` dosyası üzerinden `AI_PROVIDER` değeri yeni sağlayıcı adı ile değiştirilir.
