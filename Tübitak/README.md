# 🛡️ Violence Guard: AI & Thermal Sensor Fusion Security System

**Violence Guard**, TÜBİTAK 2209-A kapsamında geliştirilen; optik görüntü işleme (Computer Vision) ve termal görüntülemeyi birleştirerek potansiyel şiddet olaylarını, silahlı tehditleri ve anormal hareketleri tespit eden hibrit bir güvenlik sistemidir.

Bu proje, sadece nesne tespiti yapmakla kalmaz; **vücut sıcaklığı analizi** ve **hareket yoğunluğu** verilerini birleştirerek ("Sensor Fusion") yanlış alarmları minimize eder.

## 🚀 Özellikler

* **Multimodal Algılama:** RGB kamera ve MLX90640 Termal Sensör verilerinin füzyonu.
* **AI Destekli Tehdit Analizi:** Roboflow Inference API kullanılarak silah ve kavga tespiti.
* **Fizyolojik Analiz:** Kişilerin vücut sıcaklığındaki ani artışları (Adrenalin/Stres belirtisi) takip eder.
* **Akıllı Karar Mekanizması:** Sadece nesneye değil, hareket yoğunluğuna ve termal veriye dayalı "Tehdit Puanı" hesaplar.
* **Otomatik Kanıt Toplama:** Olay anında otomatik ekran görüntüsü (Screenshot) alır.

## 🛠️ Donanım Gereksinimleri

* **İşlemci:** NVIDIA Jetson Nano / Raspberry Pi 4 veya Laptop
* **Termal Sensör:** MLX90640 (I2C Arayüzü)
* **Kamera:** Standart USB Webcam

## ⚙️ Kurulum ve Çalıştırma

1.  Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install opencv-python numpy inference-sdk python-dotenv
    ```

2.  `.env` dosyanıza API anahtarlarınızı girin.

3.  Sistemi başlatın:
    ```bash
    python violence_guard.py
    ```

## 🧠 Nasıl Çalışır?

Sistem 3 veriyi birleştirir:
1.  **Görsel Tehdit:** Yapay Zeka silah görüyor mu?
2.  **Hareket Analizi:** Ortamda ani bir kaos var mı?
3.  **Termal Anomali:** Vücut sıcaklığı aniden yükselen (stres/efor) biri var mı?

Bu üç veri birleşip bir **Tehdit Skoru** oluşturur. Skor **85'i** geçerse sistem alarm verir.

---
*Developed by Arda Can Tunç within the scope of TÜBİTAK 2209-A Research Projects.*