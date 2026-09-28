import os
import threading
import time
import cv2
import numpy as np
import requests
import torch
import urllib.request
from flask import Flask, Response
from ultralytics import YOLO

# ==========================================
# ⚙️ KONFIGURASI TELEGRAM & ESP32-CAM
# ==========================================
TELEGRAM_TOKEN = "8852316590:AAHHeDhitPzzPIgVSCMlX8t-u_8DECNkfH8"
TELEGRAM_CHAT_ID = "6169828628"

# 🌐 IP ESP32-CAM (Ganti dengan link Ngrok publik jika server di Cloud, atau IP lokal jika satu WiFi)
ESP32_STREAM_URL = "http://192.168.43.100:81/stream"

MODEL_PATH = "best.pt"
CONF_THRESHOLD = 0.85
COOLDOWN_TIME = 60

# ==========================================
# 🧪 KAMUS PENANGANAN (TEKS POLOS / PLAIN TEXT)
# ==========================================
PENANGANAN_PENYAKIT = {
    "leaf spot": (
        "Tindakan Perbaikan (Fungisida):\n"
        "• Semprot fungisida kontak Mankozeb 80% (dosis 2 gr/liter) atau sistemik Difenokonazol 250 EC (0,5 ml/liter).\n"
        "• Pangkas daun tua/gejala parah di bagian bawah dan jaga sirkulasi udara.\n"
        "• Penting: Hentikan sementara pupuk N murni (Urea/ZA) agar sel daun tidak lembek.\n\n"
        "Nutrisi Pemulihan & Kekuatan Sel:\n"
        "• Kalsium Nitrat (CN): Dosis 2 gr/liter air (semprot) untuk mempertebal dinding sel daun.\n"
        "• Pupuk Mikro Kelat (Fe, Zn, Mn): Dosis 1 gr/liter air untuk mempercepat re-klorofil/fotosintesis.\n"
        "• Asam Amino: Dosis 1 ml/liter air untuk memulihkan stres jaringan akibat infeksi."
    ),
    "leaf curl": (
        "Tindakan Perbaikan (Pengendalian Vektor):\n"
        "• Semprot insektisida Abamektin 18 EC (dosis 0,75 ml/liter) atau Imidakloprid 200 SL (0,5 ml/liter) untuk membasmi Hama Thrips/Mite.\n"
        "• Pangkas pucuk tanaman yang keriting parah jika infeksi masih di tahap awal.\n\n"
        "Nutrisi Pemulihan Tunas Baru:\n"
        "• Asam Amino + Ekstrak Rumput Laut: Semprot dosis 1,5 ml/liter air setiap 5 hari untuk merangsang tumbuhnya tunas baru.\n"
        "• Pupuk MKP (Mono Kalium Phosphate): Kocor 3 gr/liter air untuk memperkuat struktur batang & jaringan pembuluh."
    ),
    "whitefly": (
        "Tindakan Perbaikan (Pengendalian Hama):\n"
        "• Semprot insektisida sistemik Imidakloprid 200 SL (dosis 0,5–1 ml/liter) diselingi dengan Minyak Nabati/Sabun Kalium sebagai pelekat.\n"
        "• Pasang Yellow Sticky Trap (Perangkap Kuning) sebanyak 20–40 buah per hektar di sekitar area tanam.\n\n"
        "Nutrisi Pencegahan Virus (Imunitas Tanaman):\n"
        "• Unsur Mikro Kelat (Zn, Fe, Cu): Dosis 1 gr/liter air (semprot) untuk menjaga imunitas dan klorofil daun.\n"
        "• Kalsium Nitrat: Dosis 2 gr/liter air untuk menebalkan jaringan daun agar tidak mudah ditusuk mulut hama (sucking insects)."
    ),
    "yellowish": (
        "Tindakan Perbaikan (Evaluasi Media & Vektor):\n"
        "• Cek drainase media tanam (jangan terlalu becek/tergenang) dan cek apakah terdapat serangan kutu di balik daun.\n"
        "• Jika disebabkan pH tanah asam, taburi Kapur Pertanian/Dolomit (100 gr/tanaman).\n\n"
        "Nutrisi Khusus Anti-Klorosis:\n"
        "• Magnesium Sulfat (MgSO4) & Unsur Besi (Fe-EDTA): Semprot dosis 1–2 gr/liter air untuk memulihkan zat hijau daun (klorofil).\n"
        "• Asam Humat: Kocor dosis 2 gr/liter air ke area perakaran untuk memaksimalkan penyerapan hara oleh akar."
    ),
}

last_sent_time = {}

# Hardware acceleration check
print("CUDA available:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load Model YOLO
model = YOLO(MODEL_PATH)
model.to(device)

DATASET_CLASSES = list(model.names.values())
print("Daftar label resmi dataset:", DATASET_CLASSES)


# ==========================================
# 📲 FUNGSI TELEGRAM BOT
# ==========================================
def send_telegram_alert(label, confidence, frame):
  current_time = time.time()

  if label in last_sent_time:
    if current_time - last_sent_time[label] < COOLDOWN_TIME:
      return

  last_sent_time[label] = current_time

  temp_img_path = f"temp_{int(current_time)}.jpg"
  cv2.imwrite(temp_img_path, frame)

  clean_label_key = label.lower().strip()
  solusi_default = (
      "Tindakan Perbaikan:\n"
      "• Pangkas bagian yang terinfeksi parah dan isolasi tanaman.\n"
      "• Hentikan sementara pupuk N kimia dosis tinggi.\n\n"
      "Nutrisi Pemulihan:\n"
      "• Semprotkan Asam Amino (1 ml/l), Unsur Mikro Kelat (1 gr/l), dan Kalsium Nitrat (2 gr/l) untuk pemulihan jaringan."
  )

  solusi = PENANGANAN_PENYAKIT.get(clean_label_key, solusi_default)

  caption = (
      f"🚨 DETEKSI PENYAKIT CABAI 🚨\n\n"
      f"📌 Kategori: {label}\n"
      f"🎯 Akurasi: {confidence:.2f}\n"
      f"⏰ Waktu: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
      f"💡 PANDUAN NUTRISI & DOSIS PEMULIHAN:\n{solusi}"
  )

  def worker():
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    try:
      with open(temp_img_path, "rb") as photo:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "caption": caption,
        }
        files = {"photo": photo}
        response = requests.post(url, data=payload, files=files, timeout=5)

        if response.status_code == 200:
          print(f"[TELEGRAM] Alert terkirim untuk: {label}")
        else:
          print(f"[TELEGRAM FAILED] {response.text}")
    except Exception as e:
      print(f"[ERROR TELEGRAM] {e}")
    finally:
      if os.path.exists(temp_img_path):
        os.remove(temp_img_path)

  threading.Thread(target=worker, daemon=True).start()


# ==========================================
# 🎥 DETEKSI & STREAMING FLASK
# ==========================================
def generate_frames():
  url = ESP32_STREAM_URL
  stream = None
  bytes_data = b""

  while True:
    if stream is None:
      print("[INFO] Menghubungkan ke ESP32-CAM...")
      try:
        stream = urllib.request.urlopen(url, timeout=5)
      except Exception as e:
        print(f"[WARNING] Gagal terhubung ke stream: {e}. Mencoba ulang...")
        time.sleep(2)
        continue

    try:
      bytes_data += stream.read(1024)
      a = bytes_data.find(b"\xff\xd8")
      b = bytes_data.find(b"\xff\xd9")

      if a != -1 and b != -1:
        jpg = bytes_data[a : b + 2]
        bytes_data = bytes_data[b + 2 :]

        frame = cv2.imdecode(
            np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
        )

        if frame is None:
          continue

        clean_frame = frame.copy()
        frame_h, frame_w, _ = frame.shape

        results = model.predict(
            source=frame, device=device, conf=CONF_THRESHOLD, verbose=False
        )

        plant_detected = False

        for r in results:
          for box in r.boxes:
            conf = float(box.conf[0])
            label = model.names[int(box.cls[0])]

            if label in DATASET_CLASSES:
              x1, y1, x2, y2 = map(int, box.xyxy[0])

              box_area = (x2 - x1) * (y2 - y1)
              frame_area = frame_w * frame_h
              if box_area > (0.7 * frame_area):
                continue

              plant_detected = True

              is_healthy = label.lower().strip() in [
                  "cabai sehat",
                  "dauncabai sehat",
                  "sehat",
              ]
              box_color = (0, 255, 0) if is_healthy else (0, 0, 255)

              cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
              cv2.putText(
                  frame,
                  f"{label} {conf:.2f}",
                  (x1, y1 - 10),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.7,
                  box_color,
                  2,
              )

              if not is_healthy:
                send_telegram_alert(label, conf, clean_frame)

        if not plant_detected:
          cv2.putText(
              frame,
              "Tidak Terdeteksi",
              (30, 50),
              cv2.FONT_HERSHEY_SIMPLEX,
              1.0,
              (0, 0, 255),
              2,
              cv2.LINE_AA,
          )

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
          continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    except Exception as e:
      print(f"[WARNING] Koneksi terputus: {e}. Reconnecting...")
      stream = None
      bytes_data = b""
      time.sleep(2)


# ==========================================
# 🌐 ROUTING SERVER FLASK (Dioptimalkan untuk Mobile/APK)
# ==========================================
app = Flask(__name__)


@app.route("/")
def index():
  return """
    <html>
        <head>
            <title>Detektor Penyakit Cabai</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { 
                    margin: 0; 
                    padding: 0; 
                    background-color: #121212; 
                    color: white; 
                    font-family: Arial, sans-serif; 
                    text-align: center; 
                }
                h2 { 
                    margin: 15px 0 10px 0; 
                    font-size: 20px; 
                    color: #4CAF50; 
                }
                .container {
                    width: 100%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
                img { 
                    width: 95%; 
                    max-width: 600px; 
                    height: auto; 
                    border: 2px solid #4CAF50; 
                    border-radius: 8px; 
                }
            </style>
        </head>
        <body>
            <h2>🌿 Live YOLOv8 Detector</h2>
            <div class="container">
                <img src="/video_feed">
            </div>
        </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
  return Response(
      generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
  )


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000, debug=False)
