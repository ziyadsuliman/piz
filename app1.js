import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ---------------------------------------------------------------------------
// عناصر DOM
// ---------------------------------------------------------------------------
const video = document.getElementById("webcam");
const cameraCanvas = document.getElementById("cameraCanvas");
const drawCanvas = document.getElementById("drawCanvas");
const cursorCanvas = document.getElementById("cursorCanvas");

const cameraCtx = cameraCanvas.getContext("2d");
const drawCtx = drawCanvas.getContext("2d");
const cursorCtx = cursorCanvas.getContext("2d");

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const fpsStat = document.getElementById("fpsStat");
const handStat = document.getElementById("handStat");
const loadingOverlay = document.getElementById("loadingOverlay");
const loaderText = document.getElementById("loaderText");
const loaderRetry = document.getElementById("loaderRetry");
const hint = document.getElementById("hint");

const clearBtn = document.getElementById("clearBtn");
const downloadBtn = document.getElementById("downloadBtn");
const brushSizeInput = document.getElementById("brushSize");
const swatches = Array.from(document.querySelectorAll(".swatch"));

// ---------------------------------------------------------------------------
// الحالة
// ---------------------------------------------------------------------------
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

const PINCH_ON = 0.55; // نسبة إغلاق القرصة (ابدأ الرسم)
const PINCH_OFF = 0.7; // نسبة فتح القرصة (ارفع القلم) - هستيرة لتفادي الرجفة
const FIST_RATIO = 1.35; // عتبة اكتشاف القبضة المغلقة
const FIST_HOLD_MS = 650; // مدة إبقاء القبضة عشان يمسح اللوحة

let activeColor = "#FFB020";
let activeSize = 10;
let handLandmarker = null;
let running = false;
let lastVideoTime = -1;
let fistStartedAt = null;
let hintShown = true;

// حالة كل يد (حتى يدين): موضع آخر نقطة رسم + هل كانت تقرص
const handState = [
  { drawing: false, lastX: null, lastY: null },
  { drawing: false, lastX: null, lastY: null },
];

// عدّاد FPS بسيط (بدون تخصيص مصفوفة كل إطار)
let frameCount = 0;
let lastFpsUpdate = performance.now();
const useFrameCallback = "requestVideoFrameCallback" in HTMLVideoElement.prototype;

// تشغيل نموذج التعرف كل N فريم فقط (يقلل الضغط على الجهاز)
const DETECT_EVERY_N_FRAMES = 2;
let frameSkipCounter = 0;
let lastResults = { landmarks: [], handedness: [] };

// ---------------------------------------------------------------------------
// تهيئة الواجهة (الألوان + حجم الفرشاة + الأزرار)
// ---------------------------------------------------------------------------
swatches.forEach((btn) => {
  btn.addEventListener("click", () => {
    swatches.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    activeColor = btn.dataset.color;
  });
});

brushSizeInput.addEventListener("input", (e) => {
  activeSize = Number(e.target.value);
});

clearBtn.addEventListener("click", clearCanvas);

downloadBtn.addEventListener("click", () => {
  const out = document.createElement("canvas");
  out.width = drawCanvas.width;
  out.height = drawCanvas.height;
  const ctx = out.getContext("2d");
  ctx.fillStyle = "#0D0F12";
  ctx.fillRect(0, 0, out.width, out.height);
  ctx.drawImage(drawCanvas, 0, 0);

  const link = document.createElement("a");
  link.download = `gesturedraw-${Date.now()}.png`;
  link.href = out.toDataURL("image/png");
  link.click();
});

loaderRetry.addEventListener("click", () => {
  loaderRetry.hidden = true;
  loaderText.textContent = "جاري إعادة المحاولة…";
  init();
});

function clearCanvas() {
  drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
}

function setStatus(text, mode) {
  statusText.textContent = text;
  statusDot.className = "status-dot" + (mode ? ` ${mode}` : "");
}

function showLoaderError(message) {
  loaderText.textContent = message;
  loaderRetry.hidden = false;
}

// ---------------------------------------------------------------------------
// إعداد الكاميرا
// ---------------------------------------------------------------------------
async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      video.play();
      resolve(video);
    };
  });
}

function resizeCanvases() {
  const w = video.videoWidth;
  const h = video.videoHeight;
  [cameraCanvas, drawCanvas, cursorCanvas].forEach((c) => {
    c.width = w;
    c.height = h;
  });
}

// ---------------------------------------------------------------------------
// تحميل نموذج تتبع اليد
// ---------------------------------------------------------------------------
async function loadModel() {
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.6,
  });
}

// ---------------------------------------------------------------------------
// حسابات هندسية على نقاط اليد (landmarks)
// ---------------------------------------------------------------------------
function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}

function pinchRatio(lm) {
  // مسافة الإبهام-السبابة مقسومة على حجم الكف (wrist -> middle_mcp)
  const handSize = dist(lm[0], lm[9]) || 1e-6;
  return dist(lm[4], lm[8]) / handSize;
}

function fistRatio(lm) {
  const handSize = dist(lm[0], lm[9]) || 1e-6;
  const tips = [8, 12, 16, 20];
  const avg =
    tips.reduce((sum, i) => sum + dist(lm[i], lm[0]), 0) / tips.length;
  return avg / handSize;
}

// ---------------------------------------------------------------------------
// حلقة الرسم الرئيسية
// ---------------------------------------------------------------------------
function drawCameraFrame() {
  cameraCtx.save();
  cameraCtx.translate(cameraCanvas.width, 0);
  cameraCtx.scale(-1, 1); // مرآة
  cameraCtx.drawImage(video, 0, 0, cameraCanvas.width, cameraCanvas.height);
  cameraCtx.restore();
  // تعتيم خفيف بدل CSS filter (أرخص بكثير على الأداء)
  cameraCtx.fillStyle = "rgba(13, 15, 18, 0.45)";
  cameraCtx.fillRect(0, 0, cameraCanvas.width, cameraCanvas.height);
}

function toScreen(lm) {
  // تحويل الإحداثيات المعيارية (0..1) إلى بكسلات مع عكس أفقي (مرآة)
  return {
    x: (1 - lm.x) * cursorCanvas.width,
    y: lm.y * cursorCanvas.height,
  };
}

function handleFist(anyFist) {
  if (anyFist) {
    if (fistStartedAt === null) fistStartedAt = performance.now();
    const held = performance.now() - fistStartedAt;
    if (held > FIST_HOLD_MS) {
      clearCanvas();
      setStatus("تم مسح اللوحة", "tracking");
      fistStartedAt = null;
    }
  } else {
    fistStartedAt = null;
  }
}

function processResults(results) {
  cursorCtx.clearRect(0, 0, cursorCanvas.width, cursorCanvas.height);

  const numHands = results.landmarks.length;
  handStat.textContent = `أيادٍ: ${numHands}`;

  let anyFist = false;

  for (let i = 0; i < 2; i++) {
    const lm = results.landmarks[i];
    const state = handState[i];

    if (!lm) {
      state.drawing = false;
      state.lastX = null;
      state.lastY = null;
      continue;
    }

    const pRatio = pinchRatio(lm);
    const fRatio = fistRatio(lm);
    if (fRatio < FIST_RATIO) anyFist = true;

    const pinching = state.drawing
      ? pRatio < PINCH_OFF
      : pRatio < PINCH_ON;

    const tip = toScreen(lm[8]);
    const thumb = toScreen(lm[4]);
    const cx = (tip.x + thumb.x) / 2;
    const cy = (tip.y + thumb.y) / 2;

    // مؤشر الإصبع
    cursorCtx.beginPath();
    cursorCtx.arc(cx, cy, pinching ? activeSize / 2 + 4 : 8, 0, Math.PI * 2);
    cursorCtx.fillStyle = pinching ? activeColor : "rgba(232,230,225,0.5)";
    cursorCtx.globalAlpha = pinching ? 0.9 : 0.6;
    cursorCtx.fill();
    cursorCtx.globalAlpha = 1;
    if (!pinching) {
      cursorCtx.lineWidth = 1.5;
      cursorCtx.strokeStyle = "rgba(232,230,225,0.8)";
      cursorCtx.stroke();
    }

    if (pinching) {
      if (hintShown) {
        hint.classList.add("hidden");
        hintShown = false;
      }
      if (state.drawing && state.lastX !== null) {
        drawCtx.beginPath();
        drawCtx.moveTo(state.lastX, state.lastY);
        drawCtx.lineTo(cx, cy);
        drawCtx.strokeStyle = activeColor;
        drawCtx.lineWidth = activeSize;
        drawCtx.lineCap = "round";
        drawCtx.lineJoin = "round";
        drawCtx.stroke();
      } else {
        // نقطة بداية الخط (نقرة سريعة ترسم نقطة)
        drawCtx.beginPath();
        drawCtx.arc(cx, cy, activeSize / 2, 0, Math.PI * 2);
        drawCtx.fillStyle = activeColor;
        drawCtx.fill();
      }
      state.drawing = true;
      state.lastX = cx;
      state.lastY = cy;
    } else {
      state.drawing = false;
      state.lastX = null;
      state.lastY = null;
    }
  }

  handleFist(anyFist);
  setStatus(
    numHands > 0 ? "تتبّع اليد نشط" : "أظهر يدك للكاميرا",
    numHands > 0 ? "tracking" : "ready"
  );
}

function runFrame() {
  drawCameraFrame();

  frameSkipCounter++;
  if (frameSkipCounter >= DETECT_EVERY_N_FRAMES) {
    frameSkipCounter = 0;
    lastResults = handLandmarker.detectForVideo(video, performance.now());
  }
  processResults(lastResults);

  frameCount++;
  const now = performance.now();
  if (now - lastFpsUpdate >= 500) {
    fpsStat.textContent = `FPS ${Math.round((frameCount * 1000) / (now - lastFpsUpdate))}`;
    frameCount = 0;
    lastFpsUpdate = now;
  }
}

function tick() {
  // مسار احتياطي للمتصفحات اللي ما تدعم requestVideoFrameCallback
  if (!running) return;
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    runFrame();
  }
  requestAnimationFrame(tick);
}

function videoFrameLoop() {
  if (!running) return;
  runFrame();
  video.requestVideoFrameCallback(videoFrameLoop);
}

// ---------------------------------------------------------------------------
// التهيئة العامة
// ---------------------------------------------------------------------------
async function init() {
  try {
    setStatus("جاري تشغيل الكاميرا…");
    await setupCamera();
    resizeCanvases();

    loaderText.textContent = "جاري تحميل نموذج تتبع اليد…";
    await loadModel();

    loadingOverlay.classList.add("hidden");
    setStatus("جاهز — أظهر يدك للكاميرا", "ready");

    running = true;
    if (useFrameCallback) {
      video.requestVideoFrameCallback(videoFrameLoop);
    } else {
      requestAnimationFrame(tick);
    }
  } catch (err) {
    console.error(err);
    setStatus("تعذّر التشغيل", "");
    if (err && err.name === "NotAllowedError") {
      showLoaderError("تم رفض إذن الكاميرا. فعّله من إعدادات المتصفح وأعد المحاولة.");
    } else {
      showLoaderError("تعذّر تحميل نموذج تتبع اليد. تحقق من اتصال الإنترنت وأعد المحاولة.");
    }
  }
}

window.addEventListener("resize", () => {
  if (video.videoWidth) resizeCanvases();
});

init();