# PulseNova 🩺✨  
**AI-Powered Medical Assistant built with Amazon Nova (Amazon Bedrock)**

PulseNova is a multimodal healthcare assistant designed to give users **clarity for their health concerns**. It combines **AI symptom triage, voice interaction, pulse monitoring, medical image analysis, lab report translation, and nearby care discovery** in one seamless web experience.

Built for the **Amazon Nova AI Hackathon**.

---

## 🚀 Features

### 1) Smart Symptom Triage (Amazon Nova)
- Chat-based triage assistant for symptom guidance
- Accepts **text input** and optional **image input**
- Provides structured responses with practical next steps
- Designed for **first-line guidance**, not diagnosis

### 2) Voice Mode (Hands-Free AI Assistant)
- Full-screen voice conversation overlay
- Voice state machine:
  - `idle`
  - `listening`
  - `thinking`
  - `speaking`
- Speech-to-text + text-to-speech
- **Barge-in support** (interrupt AI while it is speaking)

### 3) Vitals (Pulse Monitor)
- Estimates pulse using the device camera (PPG-style approach)
- Real-time waveform graph
- Signal quality indicator
- Stable BPM detection + auto stop
- Processing happens **locally in the browser** (video is not uploaded)

### 4) X-Ray / Scan Analysis (Amazon Nova Vision)
- Upload X-ray / MRI / scan images
- AI-generated summary and findings
- Structured output with next steps and disclaimer

### 5) Lab Report Translator (Amazon Nova Vision)
- Upload lab report images
- Converts complex results into plain English
- Highlights key findings, visible flags, and suggested next steps

### 6) Find Care Nearby
- Search by ZIP / city / address
- “Near Me” using geolocation
- Provider types:
  - Hospital
  - Urgent Care
  - ER
  - Pharmacy
  - Dentist
  - Primary Care
  - Pediatrician
- Distance sorting + map/directions links

---

## 🧠 Amazon Nova Integration

PulseNova uses **Amazon Nova via Amazon Bedrock** for both:

- **Text intelligence** → symptom triage chat
- **Vision intelligence** → scan analysis + lab report interpretation

### API Endpoints used by the frontend
- `POST /api/triage` → text + optional image triage
- `POST /api/vision` → image analysis (X-ray / labs)

> Amazon Nova is the core intelligence layer that powers PulseNova’s multimodal healthcare workflows.

---

## 🏗️ Tech Stack

### Frontend
- **HTML5**
- **Tailwind CSS (CDN)**
- **Lucide Icons**
- **Vanilla JavaScript** (single-page app architecture)

### Browser APIs
- `fetch`
- `FileReader`
- `SpeechRecognition / webkitSpeechRecognition`
- `SpeechSynthesis`
- `MediaDevices.getUserMedia`
- `Canvas API`
- `Geolocation API`

### Backend
- **FastAPI (Python)** *(backend API integration layer)*
- Amazon Bedrock runtime calls (Nova models)

### External Services
- **Amazon Bedrock (Amazon Nova)**
- **Google Maps JavaScript API**
- **Google Places API**

---

## 📸 Screenshots

> Add your screenshots here after uploading them to the repo.

### Home
![Home](./assets/home.png)

### Triage Assistant
![Triage](./assets/triage.png)

### Voice Mode
![Voice Mode](./assets/voice-mode.png)

### Pulse Monitor
![Vitals](./assets/vitals.png)

### X-Ray Analysis
![X-Ray](./assets/xray.png)

### Lab Translator
![Labs](./assets/labs.png)

### Find Care
![Care Finder](./assets/care.png)

---

## ⚙️ Project Structure (Suggested)

```bash
PulseNova/
├── index.html              # Main single-page frontend
├── server.py               # Backend server (FastAPI / API routes)
├── static/                 # Optional static assets
├── assets/                 # Screenshots, demo media, icons
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not committed)
└── README.md
