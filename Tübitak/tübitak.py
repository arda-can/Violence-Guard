
import os
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import time
import threading
import signal
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import deque

from urllib.request import Request, urlopen
from urllib.parse import urlencode
from urllib.error import URLError

import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient

# OpenCV log sessize alma (sürüme göre değişebilir)
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    try:
        cv2.setLogLevel(0)
    except Exception:
        pass

# Termal import (cihaz yoksa program patlamasın)
try:
    import board
    import busio
    import adafruit_mlx90640
    TERMAL_VAR = True
except Exception as e:
    TERMAL_VAR = False
    TERMAL_HATA = e

# =========================================================
# AYARLAR
# =========================================================
PENCERE_ADI = "FUSION GUARD (TERMAL + RGB + ROBOFLOW)"

# Ayna ayarları
AYNA_RGB_GORUNUM = True
AYNA_RGB_INFER = False
AYNA_TERMAL_GORUNUM = False

# Roboflow
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "").strip()
MODEL_ID = "tubitak-z4jes/1"
RF_API_URL = "https://serverless.roboflow.com"

RF_HER_N_KAREDE = 6
RF_CONF_ESIK = 0.25
DETEKSIYON_MAX_YAS_SN = 3.0
RF_INFER_W, RF_INFER_H = 768, 432

# Kamera
RGB_KAMERA_INDEX = 0
RGB_W, RGB_H = 1280, 720
GORUNTU_W, GORUNTU_H = 1280, 720
GORUNUM_OLCEK = 0.70

# Termal
I2C_FREKANS = 400000
TERMAL_W, TERMAL_H = 32, 24

# Termal "insan" blob eşiği: ABS + median+delta
INSAN_ABS_MIN_C = 28.0
INSAN_DELTA_C = 2.5

# İnsan blob filtreleri
MIN_INSAN_PIKSEL = 30
MAX_INSAN_PIKSEL = 520
MIN_BBOX_W = 4
MIN_BBOX_H = 7
ASPECT_MIN = 0.7
ASPECT_MAX = 4.2
MIN_DOLULUK_ORANI = 0.20

# "insan" kararı zamansal olsun
INSAN_STREAK_ON = 3
INSAN_STREAK_OFF = 1

# Termal kişi sayısı
MAX_TERMAL_KISI = 2

# Termal sıcaklık ölçümü: blob içindeki en sıcak Top-K ortalaması
INSAN_TOPK = 20

# Core tahmini (yüzey + offset) + kalibrasyon
CORE_TAHMIN_OFFSET_C = 4.0
KALIB_DOSYA = "thermal_calib.txt"
KALIB_HEDEF_CORE_C = 36.5

# Baseline/EMA
BASELINE_ALPHA = 0.02
TEMP_EMA_ALPHA = 0.35

# Delta (ani artış) ve "yüksek vücut sıcaklığı" (bilgi amaçlı)
ADRENALIN_DELTA_CORE_C = 1.2
ATES_MIN_C = 38.0
ATES_MAX_C = 42.5   # bunun üstü çoğu zaman "sıcak nesne/alev" gibi davranır

# Termal aşırı sıcak nesne eşiği (piksel bazlı)
ANOMALI_TEMP_C = 55.0
ANOMALI_MIN_PIKSEL = 6

TEHDIT_TUT_SN = 2.0

# Roboflow sınıfları
KISI_SINIFLARI = {"person", "human"}

TEHLIKE_SINIFLARI = {
    "knife", "gun", "weapon", "pistol", "rifle",
    "violence", "fight", "attack",
    "kavga", "siddet", "şiddet",
    "risk", "danger", "threat", "violent", "assault",
    "bıçak", "silah"
}

OTO_TEHLIKE_FALLBACK = True
OTO_TEHLIKE_MIN_CONF = 0.70

# Hareket (motion)
HAREKET_ORAN_MAX = 0.14
HAREKET_MIN_SKOR = 18
HAREKET_EMA_ALPHA = 0.25

# Tehlike EMA
TEHLIKE_EMA_ALPHA = 0.35

# Skor ağırlıkları
TEHLIKE_SKOR_CARPAN = 100            # tehlike_ema * 100 => 0..100
HAREKET_BONUS_MAX = 12
HAREKET_BONUS_CARPAN = 0.30
HAREKET_BONUS_GEREKLI_TEHLIKE = 0.18

# ✅ Kişi sayısı arttıkça hareket etkisi hafif kırılsın
# 1 kişi: 1.00 | 2 kişi: ~0.80 | 3 kişi: ~0.67 | 4 kişi: ~0.57
KISI_BASINA_HAREKET_KIRMA = 0.25

# ✅ Ekranda "TEHDİT: EVET" yazması için minimum skor (SS eşiklerinden bağımsız)
TEHDIT_GORSEL_ESIK = 30

# Screenshot
SS_KLASOR = "ssler"
SS_ESIK = 85
SS_COOLDOWN_SN = 2.0
SS_OLCEK = 0.75
SS_MAX_DOSYA = 40

SS_GORUNTU_DONDUR_SN = 1.0
SS_BANNER_SN = 1.0

# KAVGA ONAY KURALI
ONAY_PENCERE_SN = 12.0
ONAY_MIN_SS = 5
ONAY_COOLDOWN_SN = 30.0
ONAY_BANNER_SN = 10.0

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_TIMEOUT_SN = 5

ALARM_LOG_DOSYA = "alerts.log"

# =========================================================
# VERI YAPILARI
# =========================================================
@dataclass
class Algilama:
    sinif: str
    guven: float
    x1: int
    y1: int
    x2: int
    y2: int
    src_w: int
    src_h: int

@dataclass
class TermalKisi:
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    alan: int
    oran: float
    doluluk: float
    temp_yuzey: float
    temp_core: float
    delta_core: float
    streak: int

@dataclass
class _Iz:
    track_id: int
    cx: float
    cy: float
    temp_core_ema: float
    base_core_ema: float
    son_gorulme: float
    streak: int
    bbox: Tuple[int, int, int, int]
    alan: int
    oran: float
    doluluk: float

# =========================================================
# PAYLASILAN DURUMLAR
# =========================================================
dur_event = threading.Event()

termal_lock = threading.Lock()
termal_img: Optional[np.ndarray] = None
termal_kisiler: List[TermalKisi] = []

# ✅ iki farklı durum:
termal_sicak_nesne: bool = False          # >=55C sıcak piksel var mı? (insan dahil)
termal_insan_disi_sicak: bool = False     # >=55C sıcak piksel insan dışında var mı?
termal_hot_px_total: int = 0              # >=55C piksel sayısı (toplam)
termal_hot_px_dis: int = 0                # >=55C piksel sayısı (insan dışı)

termal_istatistik: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
termal_thr: float = 0.0
sicaklik_offset_c: float = 0.0

det_lock = threading.Lock()
son_dets: List[Algilama] = []
son_det_zaman: float = 0.0
son_top: List[Tuple[str, float]] = []

# =========================================================
# YARDIMCILAR
# =========================================================
def _topk_ort(vals: np.ndarray, k: int) -> float:
    if vals.size == 0:
        return 0.0
    k = max(1, min(int(k), int(vals.size)))
    topk = np.partition(vals, -k)[-k:]
    return float(np.mean(topk))

def _kalib_yukle() -> float:
    try:
        if os.path.exists(KALIB_DOSYA):
            with open(KALIB_DOSYA, "r", encoding="utf-8") as f:
                return float(f.read().strip())
    except Exception:
        pass
    return 0.0

def _kalib_kaydet(offset: float) -> None:
    try:
        with open(KALIB_DOSYA, "w", encoding="utf-8") as f:
            f.write(f"{offset:.4f}")
    except Exception:
        pass

def _ss_temizle_sonN(klasor: str, tut: int) -> None:
    """Klasörde sadece son `tut` adet .png kalsın (en eskileri siler)."""
    try:
        dosyalar = []
        for ad in os.listdir(klasor):
            if ad.lower().endswith(".png"):
                yol = os.path.join(klasor, ad)
                try:
                    dosyalar.append((os.path.getmtime(yol), yol))
                except Exception:
                    pass
        dosyalar.sort(key=lambda x: x[0])  # eski -> yeni
        while len(dosyalar) > tut:
            _, yol = dosyalar.pop(0)
            try:
                os.remove(yol)
            except Exception:
                break
    except Exception:
        pass

# =========================================================
# TELEGRAM
# =========================================================
def _telegram_aktif_mi() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

def _tg_post(method: str, fields: dict, file_field: Optional[str] = None, file_path: Optional[str] = None) -> bool:
    if not _telegram_aktif_mi():
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"

    try:
        if file_field and file_path:
            boundary = "----FUSION_GUARD_BOUNDARY_" + str(int(time.time() * 1000))
            parts = []

            def add_text(name: str, val: str):
                parts.append(
                    (f"--{boundary}\r\n"
                     f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                     f"{val}\r\n").encode("utf-8")
                )

            for k, v in fields.items():
                add_text(str(k), str(v))

            fname = os.path.basename(file_path)
            parts.append(
                (f"--{boundary}\r\n"
                 f'Content-Disposition: form-data; name="{file_field}"; filename="{fname}"\r\n'
                 f"Content-Type: image/png\r\n\r\n").encode("utf-8")
            )
            with open(file_path, "rb") as f:
                parts.append(f.read())
            parts.append(b"\r\n")
            parts.append(f"--{boundary}--\r\n".encode("utf-8"))

            body = b"".join(parts)
            req = Request(url, data=body, method="POST")
            req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

        else:
            body = urlencode(fields).encode("utf-8")
            req = Request(url, data=body, method="POST")
            req.add_header("Content-Type", "application/x-www-form-urlencoded")

        with urlopen(req, timeout=TELEGRAM_TIMEOUT_SN) as resp:
            _ = resp.read()
        return True

    except URLError as e:
        print(f"[UYARI] Telegram gonderilemedi (URLError): {e}")
        return False
    except Exception as e:
        print(f"[UYARI] Telegram gonderilemedi: {e}")
        return False

def telegram_mesaj_gonder(metin: str) -> None:
    if not _telegram_aktif_mi():
        print("[BILGI] Telegram kapali (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID yok).")
        return
    _tg_post("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": metin})

def telegram_foto_gonder(png_yol: str, caption: str) -> None:
    if not _telegram_aktif_mi():
        return
    _tg_post("sendPhoto", {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}, file_field="photo", file_path=png_yol)

def _alarm_log_yaz(metin: str) -> None:
    try:
        with open(ALARM_LOG_DOSYA, "a", encoding="utf-8") as f:
            f.write(metin + "\n")
    except Exception:
        pass

def kavga_onayi_bildir(
    ss_sayisi: int, skor: int, tehlike_ema: float, hareket_skor: int,
    t_sayi: int, t_core_max: float, t_delta_max: float, ss_yol: Optional[str]
) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    metin = (
        "🚨 KAVGA ŞÜPHESİ YÜKSEK!\n"
        "Sistem olayın gerçek kavga olma ihtimalini yüksek gördü.\n\n"
        f"🕒 Zaman: {ts}\n"
        f"📸 Son {ONAY_PENCERE_SN:.0f} sn içinde alınan görüntü: {ss_sayisi} adet\n\n"
        f"⚠️ Tehdit Puanı: {skor}/100\n"
        f"🧠 Model Tehlike Güveni: %{tehlike_ema*100:.0f}\n"
        f"🏃 Hareket Seviyesi: {hareket_skor}/50\n\n"
        f"🌡️ Termal Kamera:\n"
        f"   - Kişi sayısı: {t_sayi}\n"
        f"   - En yüksek tahmini vücut sıcaklığı: {t_core_max:.1f}°C\n"
        f"   - Ani sıcaklık artışı: {t_delta_max:+.1f}°C (kişinin normaline göre)\n\n"
        "📌 Not: Bu mesaj otomatik uyarıdır. Lütfen kamera görüntüsünden durumu kontrol ediniz."
    )

    print("\n" + "=" * 80)
    print(metin)
    print("=" * 80 + "\n")
    _alarm_log_yaz(metin)

    def _gonder():
        telegram_mesaj_gonder(metin)
        if ss_yol and os.path.exists(ss_yol):
            telegram_foto_gonder(ss_yol, f"Kavga onayı | Skor={skor} | SS={ss_sayisi}")

    threading.Thread(target=_gonder, daemon=True).start()

# =========================================================
# ROBOFLOW
# =========================================================
def _rf_client() -> InferenceHTTPClient:
    if not ROBOFLOW_API_KEY:
        raise RuntimeError("ROBOFLOW_API_KEY env degiskeni bos. Anahtari env'e koymalisin.")
    return InferenceHTTPClient(api_url=RF_API_URL, api_key=ROBOFLOW_API_KEY)

def _rf_parse(result: dict, img_w: int, img_h: int) -> List[Algilama]:
    dets: List[Algilama] = []
    preds = result.get("predictions", []) or []
    for p in preds:
        sinif = str(p.get("class", "obj"))
        guven = float(p.get("confidence", 0.0))
        if guven < RF_CONF_ESIK:
            continue

        x = float(p.get("x", 0))
        y = float(p.get("y", 0))
        w = float(p.get("width", 0))
        h = float(p.get("height", 0))

        x1 = int(max(0, x - w / 2))
        y1 = int(max(0, y - h / 2))
        x2 = int(min(img_w - 1, x + w / 2))
        y2 = int(min(img_h - 1, y + h / 2))

        dets.append(Algilama(sinif=sinif, guven=guven, x1=x1, y1=y1, x2=x2, y2=y2, src_w=img_w, src_h=img_h))
    return dets

class RFWorker:
    def __init__(self):
        self.client = _rf_client()
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def submit(self, frame_bgr: np.ndarray):
        with self.lock:
            self.latest_frame = frame_bgr

    def loop(self):
        global son_dets, son_det_zaman, son_top
        while not dur_event.is_set():
            frame = None
            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame
                    self.latest_frame = None

            if frame is None:
                time.sleep(0.01)
                continue

            try:
                result = self.client.infer(frame, model_id=MODEL_ID)
                dets = _rf_parse(result, frame.shape[1], frame.shape[0])
                top = sorted([(d.sinif, d.guven) for d in dets], key=lambda x: x[1], reverse=True)[:4]

                with det_lock:
                    son_dets = dets
                    son_det_zaman = time.time()
                    son_top = top
            except Exception:
                continue

# =========================================================
# TERMAL: 2 kisi + takip + baseline
# =========================================================
def _bbox_merkez(x1: int, y1: int, x2: int, y2: int) -> Tuple[float, float]:
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def _termal_adaylari_bul(data_s: np.ndarray, thr: float) -> List[Tuple[int,int,int,int,int,float,float,np.ndarray]]:
    maske = (data_s >= thr).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(maske, connectivity=8)

    adaylar = []
    for cid in range(1, num):
        x = int(stats[cid, cv2.CC_STAT_LEFT])
        y = int(stats[cid, cv2.CC_STAT_TOP])
        w = int(stats[cid, cv2.CC_STAT_WIDTH])
        h = int(stats[cid, cv2.CC_STAT_HEIGHT])
        alan = int(stats[cid, cv2.CC_STAT_AREA])

        if w <= 0 or h <= 0:
            continue

        x1, y1, x2, y2 = x, y, x + w - 1, y + h - 1
        oran = (h / float(w)) if w > 0 else 0.0
        doluluk = alan / float(w * h)

        if not (MIN_INSAN_PIKSEL <= alan <= MAX_INSAN_PIKSEL):
            continue
        if not (w >= MIN_BBOX_W and h >= MIN_BBOX_H):
            continue
        if not (ASPECT_MIN <= oran <= ASPECT_MAX):
            continue
        if doluluk < MIN_DOLULUK_ORANI:
            continue

        comp = (labels == cid)
        adaylar.append((x1, y1, x2, y2, alan, oran, doluluk, comp))

    adaylar.sort(key=lambda x: x[4], reverse=True)
    return adaylar[:MAX_TERMAL_KISI]

def _izleri_esle(tracks: List[_Iz], adaylar, simdi: float, max_uzaklik: float = 6.0) -> None:
    kullanilan = set()

    for tr in tracks:
        en_iyi_i = None
        en_iyi_d = 1e9

        for i, (x1,y1,x2,y2,alan,oran,doluluk,comp) in enumerate(adaylar):
            if i in kullanilan:
                continue
            cx, cy = _bbox_merkez(x1,y1,x2,y2)
            d = (tr.cx - cx) ** 2 + (tr.cy - cy) ** 2
            if d < en_iyi_d:
                en_iyi_d = d
                en_iyi_i = i

        if en_iyi_i is not None and en_iyi_d <= (max_uzaklik ** 2):
            kullanilan.add(en_iyi_i)
            x1,y1,x2,y2,alan,oran,doluluk,comp = adaylar[en_iyi_i]
            cx, cy = _bbox_merkez(x1,y1,x2,y2)

            tr.cx, tr.cy = cx, cy
            tr.bbox = (x1,y1,x2,y2)
            tr.alan = alan
            tr.oran = oran
            tr.doluluk = doluluk
            tr.son_gorulme = simdi
            tr.streak = min(INSAN_STREAK_ON + 3, tr.streak + 1)
        else:
            tr.streak = max(0, tr.streak - 1)

    sonraki_id = (max([t.track_id for t in tracks], default=0) + 1)
    for i, (x1,y1,x2,y2,alan,oran,doluluk,comp) in enumerate(adaylar):
        if i in kullanilan:
            continue
        if len(tracks) >= MAX_TERMAL_KISI:
            continue

        cx, cy = _bbox_merkez(x1,y1,x2,y2)
        tracks.append(_Iz(
            track_id=sonraki_id,
            cx=cx, cy=cy,
            temp_core_ema=0.0,
            base_core_ema=0.0,
            son_gorulme=simdi,
            streak=1,
            bbox=(x1,y1,x2,y2),
            alan=alan,
            oran=oran,
            doluluk=doluluk
        ))
        sonraki_id += 1

    tracks[:] = [t for t in tracks if (simdi - t.son_gorulme) <= 1.0 or t.streak > 0]

def termal_worker():
    global (
        termal_img, termal_kisiler,
        termal_sicak_nesne, termal_insan_disi_sicak,
        termal_hot_px_total, termal_hot_px_dis,
        termal_istatistik, termal_thr, sicaklik_offset_c
    )

    if not TERMAL_VAR:
        print(f"[UYARI] Termal moduller yuklenemedi: {TERMAL_HATA}")
        return

    print("[BILGI] Termal thread basladi...")
    i2c = busio.I2C(board.SCL, board.SDA, frequency=I2C_FREKANS)
    mlx = adafruit_mlx90640.MLX90640(i2c, address=0x33)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ

    frame = np.zeros((TERMAL_W * TERMAL_H,), dtype=float)
    izler: List[_Iz] = []

    while not dur_event.is_set():
        try:
            mlx.getFrame(frame)
            data = frame.reshape((TERMAL_H, TERMAL_W)).copy()
            data_s = cv2.GaussianBlur(data, (3, 3), 0)

            tmin = float(np.min(data_s))
            tmax = float(np.max(data_s))
            tavg = float(np.mean(data_s))
            tmed = float(np.median(data_s))

            thr = max(tmed + INSAN_DELTA_C, INSAN_ABS_MIN_C)
            adaylar = _termal_adaylari_bul(data_s, thr)

            simdi = time.time()
            _izleri_esle(izler, adaylar, simdi)

            kisiler_out: List[TermalKisi] = []
            insan_maskeleri: List[np.ndarray] = []

            for tr in sorted(izler, key=lambda x: x.streak, reverse=True)[:MAX_TERMAL_KISI]:
                x1, y1, x2, y2 = tr.bbox
                x1 = max(0, min(TERMAL_W - 1, x1))
                x2 = max(0, min(TERMAL_W - 1, x2))
                y1 = max(0, min(TERMAL_H - 1, y1))
                y2 = max(0, min(TERMAL_H - 1, y2))

                roi = data_s[y1:y2+1, x1:x2+1]
                roi_mask = (roi >= thr)
                vals = roi[roi_mask] if np.any(roi_mask) else roi.flatten()
                temp_yuzey = _topk_ort(vals, INSAN_TOPK)

                with termal_lock:
                    off = float(sicaklik_offset_c)
                temp_core = temp_yuzey + CORE_TAHMIN_OFFSET_C + off

                if tr.temp_core_ema <= 0.0:
                    tr.temp_core_ema = temp_core
                else:
                    tr.temp_core_ema = (1.0 - TEMP_EMA_ALPHA) * tr.temp_core_ema + TEMP_EMA_ALPHA * temp_core

                if tr.base_core_ema <= 0.0:
                    tr.base_core_ema = tr.temp_core_ema
                else:
                    tr.base_core_ema = (1.0 - BASELINE_ALPHA) * tr.base_core_ema + BASELINE_ALPHA * tr.temp_core_ema

                delta_core = tr.temp_core_ema - tr.base_core_ema

                if tr.streak >= INSAN_STREAK_ON:
                    kisiler_out.append(TermalKisi(
                        track_id=tr.track_id,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        alan=tr.alan,
                        oran=tr.oran,
                        doluluk=tr.doluluk,
                        temp_yuzey=float(temp_yuzey),
                        temp_core=float(tr.temp_core_ema),
                        delta_core=float(delta_core),
                        streak=tr.streak
                    ))

                    full_mask = np.zeros_like(data_s, dtype=bool)
                    full_mask[y1:y2+1, x1:x2+1] = roi_mask
                    insan_maskeleri.append(full_mask)

            # ✅ Aşırı sıcak pikseller (toplam) ve insan dışı
            hot_total_mask = (data_s >= ANOMALI_TEMP_C)
            hot_px_total = int(np.sum(hot_total_mask))

            insan_union = np.zeros_like(hot_total_mask, dtype=bool)
            for m in insan_maskeleri:
                insan_union |= m

            hot_dis_mask = hot_total_mask & (~insan_union)
            hot_px_dis = int(np.sum(hot_dis_mask))

            sicak_nesne = (hot_px_total >= ANOMALI_MIN_PIKSEL)
            insan_disi_sicak = (hot_px_dis >= ANOMALI_MIN_PIKSEL)

            # görsel termal
            norm = cv2.normalize(data_s, None, 0, 255, cv2.NORM_MINMAX)
            norm = np.uint8(norm)
            img = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
            img = cv2.resize(img, (GORUNTU_W, GORUNTU_H), interpolation=cv2.INTER_LINEAR)
            if AYNA_TERMAL_GORUNUM:
                img = cv2.flip(img, 1)

            sx = GORUNTU_W / float(TERMAL_W)
            sy = GORUNTU_H / float(TERMAL_H)

            for p in kisiler_out:
                rx1, rx2 = int(p.x1 * sx), int((p.x2 + 1) * sx)
                ry1, ry2 = int(p.y1 * sy), int((p.y2 + 1) * sy)
                if AYNA_TERMAL_GORUNUM:
                    rx1, rx2 = (GORUNTU_W - rx2), (GORUNTU_W - rx1)

                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)
                cv2.putText(img, f"ID{p.track_id} core~:{p.temp_core:.1f} d:{p.delta_core:+.1f}",
                            (rx1 + 6, max(20, ry1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # ✅ Termal üst yazılar (Türkçe)
            cv2.putText(img, f"Min:{tmin:.1f}  Max:{tmax:.1f}  Ort:{tavg:.1f}  Med:{tmed:.1f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(img, f"Insan Esigi:{thr:.1f}  Kisi:{len(kisiler_out)}  SicakPxl(>={ANOMALI_TEMP_C:.0f}C):{hot_px_total}",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(img, f"Sicak Nesne: {'VAR' if sicak_nesne else 'YOK'} | Insan Disi Sicak Nokta: {'VAR' if insan_disi_sicak else 'YOK'}",
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255) if (sicak_nesne or insan_disi_sicak) else (0, 255, 0), 2)

            with termal_lock:
                termal_img = img
                termal_kisiler = kisiler_out
                termal_sicak_nesne = sicak_nesne
                termal_insan_disi_sicak = insan_disi_sicak
                termal_hot_px_total = hot_px_total
                termal_hot_px_dis = hot_px_dis
                termal_istatistik = (tmin, tmax, tavg, tmed)
                termal_thr = thr

            time.sleep(0.001)

        except ValueError:
            continue
        except Exception:
            continue

# =========================================================
# MAIN
# =========================================================
def _termal_placeholder() -> np.ndarray:
    img = np.zeros((GORUNTU_H, GORUNTU_W, 3), dtype=np.uint8)
    cv2.putText(img, "TERMAL: YOK / BAGLI DEGIL", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img

def main():
    global sicaklik_offset_c

    def sigint_yakala(sig, frame):
        dur_event.set()
    signal.signal(signal.SIGINT, sigint_yakala)

    os.makedirs(SS_KLASOR, exist_ok=True)

    sicaklik_offset_c = _kalib_yukle()
    if abs(sicaklik_offset_c) > 0.01:
        print(f"[BILGI] Kalibrasyon yuklendi: sicaklik_offset_c={sicaklik_offset_c:+.2f}C ({KALIB_DOSYA})")

    ss_zamanlari = deque()
    son_onay_zamani = 0.0

    son_ss_zaman = 0.0
    ss_id = 0

    son_ss_bildirimi = 0.0
    donmus_kare = None
    donma_bitis = 0.0

    print("[BILGI] Termal baslatiliyor...")
    if TERMAL_VAR:
        threading.Thread(target=termal_worker, daemon=True).start()
    else:
        print(f"[UYARI] Termal yok: {TERMAL_HATA}")

    print("[BILGI] Roboflow baslatiliyor...")
    rf = None
    try:
        rf = RFWorker()
    except Exception as e:
        print(f"[UYARI] Roboflow baslatilamadi: {e}")

    print("[BILGI] USB kamera baslatiliyor...")
    cap = cv2.VideoCapture(RGB_KAMERA_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RGB_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RGB_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    except Exception:
        pass

    time.sleep(1.0)

    if not cap.isOpened():
        print("[HATA] USB kamera acilamadi!")
        dur_event.set()
        return

    cv2.namedWindow(PENCERE_ADI, cv2.WINDOW_NORMAL)

    son_tehdit = 0.0
    frame_id = 0

    fps_t0 = time.time()
    fps_say = 0
    fps_val = 0.0

    onceki_gray = None
    hareket_ema = 0.0

    tehlike_ema = 0.0

    print("[BASLAT] ESC=CIKIS | K=Termal kalibrasyon (core~->36.5C)")
    print("[BILGI] Telegram:", "AKTIF" if _telegram_aktif_mi() else "KAPALI")

    while not dur_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        frame_id += 1

        fps_say += 1
        if time.time() - fps_t0 >= 1.0:
            fps_val = fps_say / (time.time() - fps_t0)
            fps_say = 0
            fps_t0 = time.time()

        infer_src = frame
        if AYNA_RGB_INFER:
            infer_src = cv2.flip(infer_src, 1)

        if rf is not None and (frame_id % RF_HER_N_KAREDE == 0):
            rf_frame = cv2.resize(infer_src, (RF_INFER_W, RF_INFER_H), interpolation=cv2.INTER_AREA)
            rf.submit(rf_frame)

        rgb = cv2.resize(frame, (GORUNTU_W, GORUNTU_H), interpolation=cv2.INTER_AREA)
        if AYNA_RGB_GORUNUM:
            rgb = cv2.flip(rgb, 1)

        # Hareket (0..50)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        hareket_oran = 0.0
        if onceki_gray is not None:
            diff = cv2.absdiff(gray, onceki_gray)
            _, th = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            degisen = float(cv2.countNonZero(th))
            hareket_oran = degisen / float(th.size)
        onceki_gray = gray

        hareket_ema = (1.0 - HAREKET_EMA_ALPHA) * hareket_ema + HAREKET_EMA_ALPHA * hareket_oran
        hareket_skor = int(min(50.0, (hareket_ema / HAREKET_ORAN_MAX) * 50.0))

        # Termal
        with termal_lock:
            t_img = None if termal_img is None else termal_img.copy()
            t_kisiler = list(termal_kisiler)
            t_sicak_nesne = bool(termal_sicak_nesne)
            t_insan_disi_sicak = bool(termal_insan_disi_sicak)

        if t_img is None:
            t_img = _termal_placeholder()

        t_sayi = len(t_kisiler)
        t_var = t_sayi > 0
        t_core_max = max((p.temp_core for p in t_kisiler), default=0.0)
        t_delta_max = max((p.delta_core for p in t_kisiler), default=0.0)

        # Roboflow
        with det_lock:
            dets = list(son_dets)
            det_yas = time.time() - son_det_zaman
        dets_gecerli = (det_yas <= DETEKSIYON_MAX_YAS_SN)

        kisi_dets = []
        tehlike_dets = []
        kisi_olmayan_max = 0.0

        if dets_gecerli and dets:
            for d in dets:
                cl = d.sinif.strip().lower()
                if cl in KISI_SINIFLARI:
                    kisi_dets.append(d)
                if cl in TEHLIKE_SINIFLARI:
                    tehlike_dets.append(d)
                if cl not in KISI_SINIFLARI:
                    kisi_olmayan_max = max(kisi_olmayan_max, d.guven)

        rgb_kisi = (len(kisi_dets) > 0)
        insan_var = (rgb_kisi or t_var)

        # TOP: sadece tehlike sınıfları (okunur olsun)
        top_tehlike = sorted([(d.sinif, d.guven) for d in tehlike_dets], key=lambda x: x[1], reverse=True)[:3]

        # tehlike_conf
        tehlike_conf = max((d.guven for d in tehlike_dets), default=0.0)

        if (OTO_TEHLIKE_FALLBACK
            and tehlike_conf == 0.0
            and kisi_olmayan_max >= OTO_TEHLIKE_MIN_CONF
            and insan_var
            and hareket_skor >= HAREKET_MIN_SKOR):
            tehlike_conf = kisi_olmayan_max

        if dets_gecerli:
            tehlike_ema = (1.0 - TEHLIKE_EMA_ALPHA) * tehlike_ema + TEHLIKE_EMA_ALPHA * tehlike_conf
        else:
            tehlike_ema *= 0.98

        # Kutu çizimleri
        if dets_gecerli and dets:
            src_w = dets[0].src_w
            src_h = dets[0].src_h
            sx = GORUNTU_W / float(src_w)
            sy = GORUNTU_H / float(src_h)

            for d in dets:
                x1 = int(d.x1 * sx); y1 = int(d.y1 * sy)
                x2 = int(d.x2 * sx); y2 = int(d.y2 * sy)
                if AYNA_RGB_GORUNUM:
                    x1, x2 = (GORUNTU_W - x2), (GORUNTU_W - x1)
                cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(rgb, f"{d.sinif} %{d.guven*100:.0f}", (x1, max(20, y1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # =================================================
        # HAREKET kişi sayısı ile hafif kırp
        # =================================================
        kisi_est = max(t_sayi, len(kisi_dets))
        kisi_est = int(max(0, min(4, kisi_est)))
        kirma = 1.0 / (1.0 + KISI_BASINA_HAREKET_KIRMA * max(0, kisi_est - 1))
        hareket_skor_eff = int(max(0, min(50, hareket_skor * kirma)))

        # =================================================
        # TEHDIT SKORU
        # =================================================
        tehlike_skor = int(tehlike_ema * TEHLIKE_SKOR_CARPAN)

        hareket_bonus = 0
        if insan_var and tehlike_ema >= HAREKET_BONUS_GEREKLI_TEHLIKE:
            hareket_bonus = int(min(HAREKET_BONUS_MAX, hareket_skor_eff * HAREKET_BONUS_CARPAN))

        # termal ipuçları (bilgi)
        ates_var = t_var and (ATES_MIN_C <= t_core_max <= ATES_MAX_C)
        delta_var = t_var and (t_delta_max >= ADRENALIN_DELTA_CORE_C)

        termal_bonus = 0
        # ✅ İnsan dışı sıcak nokta: güçlü bir "olay" göstergesi (yangın, alev, vs.)
        if t_insan_disi_sicak:
            termal_bonus += 25

        # ✅ Kavga için termal kombinasyon: 2 kişi + ani artış + hareket
        if t_var and t_sayi >= 2 and delta_var and hareket_skor_eff >= HAREKET_MIN_SKOR:
            termal_bonus += 15

        if t_var and t_sayi >= 2 and hareket_skor_eff >= (HAREKET_MIN_SKOR + 6) and tehlike_ema >= 0.20:
            termal_bonus += 10

        tehdit_skor = min(100, tehlike_skor + hareket_bonus + termal_bonus)

        # =================================================
        # TEHDIT kararı (✅ "ATEŞ tek başına" artık tehdit etmez)
        # =================================================
        tehdit_now = False
        if t_insan_disi_sicak:
            tehdit_now = True
        elif insan_var and tehlike_ema >= 0.62:
            tehdit_now = True
        elif insan_var and tehlike_ema >= 0.30 and hareket_skor_eff >= (HAREKET_MIN_SKOR + 4):
            tehdit_now = True
        elif t_var and t_sayi >= 2 and delta_var and hareket_skor_eff >= HAREKET_MIN_SKOR:
            tehdit_now = True

        if tehdit_now:
            son_tehdit = time.time()
        tehdit_aktif = (time.time() - son_tehdit) <= TEHDIT_TUT_SN

        # ✅ ekranda EVET yazması için min skor
        tehdit_goster = tehdit_aktif and (tehdit_skor >= TEHDIT_GORSEL_ESIK or t_insan_disi_sicak)

        # =================================================
        # EKRAN YAZILARI (Türkçe)
        # =================================================
        cv2.putText(rgb, f"FPS: {fps_val:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.putText(rgb, f"RGB Kisi Algilandi: {'EVET' if rgb_kisi else 'HAYIR'}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if t_var:
            cv2.putText(rgb, f"Termal Kisi: {t_sayi} | En Yuksek Tahmini Sicaklik: {t_core_max:.1f}C | Ani Artis: {t_delta_max:+.1f}C",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        else:
            cv2.putText(rgb, "Termal Kisi: 0", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.putText(
            rgb,
            f"Model Tehlike Guveni: anlik=%{tehlike_conf*100:.0f} | ort=%{tehlike_ema*100:.0f} | esik=%{RF_CONF_ESIK*100:.0f}",
            (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2
        )

        model_txt = "Model Algilama: " + (" | ".join([f"{c} %{v*100:.0f}" for c, v in top_tehlike]) if top_tehlike else "-")
        cv2.putText(rgb, model_txt, (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.putText(rgb, f"Hareket: {hareket_skor}/50 (kisi duzeltmeli: {hareket_skor_eff}/50)",
                    (20, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

        ipuclari = []
        if ates_var: ipuclari.append("YUKSEK_VUCUT_SICAKLIGI")
        if delta_var: ipuclari.append("ANI_SICAKLIK_ARTISI")
        if t_sicak_nesne: ipuclari.append("SICAK_NESNE(>=55C)")
        if t_insan_disi_sicak: ipuclari.append("INSAN_DISI_SICAK_NOKTA")
        if kisi_est >= 2: ipuclari.append("2+KISI")

        ipucu_txt = "IPUCU: " + (", ".join(ipuclari) if ipuclari else "-")
        cv2.putText(rgb, ipucu_txt, (20, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if tehdit_goster:
            cv2.putText(rgb, f"TEHDIT: EVET  skor={tehdit_skor}", (20, 315),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            cv2.putText(rgb, f"TEHDIT: HAYIR skor={tehdit_skor}", (20, 315),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # birleştir
        out = np.hstack((t_img, rgb))

        # görünüm küçült
        if GORUNUM_OLCEK != 1.0:
            out_view = cv2.resize(out,
                                  (int(out.shape[1] * GORUNUM_OLCEK), int(out.shape[0] * GORUNUM_OLCEK)),
                                  interpolation=cv2.INTER_AREA)
        else:
            out_view = out

        # =========================
        # SS Kaydet + Onay
        # =========================
        simdi = time.time()
        kaydedilen_ss_yol = None

        if tehdit_aktif and tehdit_skor >= SS_ESIK and (simdi - son_ss_zaman) >= SS_COOLDOWN_SN:
            son_ss_zaman = simdi
            ss_id += 1
            ts = time.strftime("%Y%m%d_%H%M%S")
            dosya = os.path.join(SS_KLASOR, f"ss_{ts}_{ss_id:04d}_score{tehdit_skor}.png")

            ss_img = out
            if SS_OLCEK != 1.0:
                ss_img = cv2.resize(ss_img,
                                    (int(ss_img.shape[1] * SS_OLCEK), int(ss_img.shape[0] * SS_OLCEK)),
                                    interpolation=cv2.INTER_AREA)

            cv2.imwrite(dosya, ss_img)
            _ss_temizle_sonN(SS_KLASOR, SS_MAX_DOSYA)
            kaydedilen_ss_yol = dosya
            print(f"[SS] Kaydedildi: {dosya} (son {SS_MAX_DOSYA} tutuluyor)")

            son_ss_bildirimi = simdi
            donmus_kare = out_view.copy()
            donma_bitis = simdi + SS_GORUNTU_DONDUR_SN

            ss_zamanlari.append(simdi)
            while ss_zamanlari and (simdi - ss_zamanlari[0]) > ONAY_PENCERE_SN:
                ss_zamanlari.popleft()

            if (len(ss_zamanlari) >= ONAY_MIN_SS) and ((simdi - son_onay_zamani) >= ONAY_COOLDOWN_SN):
                son_onay_zamani = simdi
                kavga_onayi_bildir(
                    ss_sayisi=len(ss_zamanlari),
                    skor=int(tehdit_skor),
                    tehlike_ema=float(tehlike_ema),
                    hareket_skor=int(hareket_skor_eff),
                    t_sayi=int(t_sayi),
                    t_core_max=float(t_core_max),
                    t_delta_max=float(t_delta_max),
                    ss_yol=kaydedilen_ss_yol
                )

        # donma efekti
        gosterilecek = out_view
        if (donmus_kare is not None) and (simdi < donma_bitis):
            gosterilecek = donmus_kare.copy()
        else:
            donmus_kare = None

        if (simdi - son_ss_bildirimi) <= SS_BANNER_SN and son_ss_bildirimi > 0:
            cv2.putText(gosterilecek, "📸 SS ALINDI", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        if (simdi - son_onay_zamani) <= ONAY_BANNER_SN and son_onay_zamani > 0:
            cv2.putText(gosterilecek, "KAVGA ONAYLANDI! (12sn icinde 5+ SS)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imshow(PENCERE_ADI, gosterilecek)

        key = (cv2.waitKey(1) & 0xFF)
        if key == 27:
            dur_event.set()
            break

        if key in (ord('k'), ord('K')):
            with termal_lock:
                ppl = list(termal_kisiler)
                off = float(sicaklik_offset_c)

            if ppl:
                current_core = float(max(p.temp_core for p in ppl))
                delta = (KALIB_HEDEF_CORE_C - current_core)
                sicaklik_offset_c = off + delta
                _kalib_kaydet(sicaklik_offset_c)
                print(f"[KALIB] core~ {current_core:.2f} -> {KALIB_HEDEF_CORE_C:.2f} | yeni offset={sicaklik_offset_c:+.2f}C")
            else:
                print("[KALIB] Termalde kisi yokken kalibrasyon yapilamaz.")

    cap.release()
    cv2.destroyAllWindows()
    dur_event.set()
    print("[CIKIS] Program kapatildi.")

if __name__ == "__main__":
    main()
