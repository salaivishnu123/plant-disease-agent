const dropZone = document.getElementById('drop-zone');
const dropZoneContent = document.getElementById('drop-zone-content');
const imageInput = document.getElementById('imageInput');
const previewImage = document.getElementById('preview-image');
const predictBtn = document.getElementById('predict-btn');
const errorMsg = document.getElementById('error-msg');
const uploadCard = document.getElementById('upload-card');
const loadingState = document.getElementById('loading-state');
const loadingText = document.getElementById('loading-text');
const resultsSection = document.getElementById('results');
const resetBtn = document.getElementById('reset-btn');

let selectedFile = null;

const LOADING_MESSAGES = [
  "Reading leaf tissue…",
  "Matching against known patterns…",
  "Weighing plant vs. disease signals…",
  "Tracing the affected region…",
];

dropZone.addEventListener('click', () => imageInput.click());

['dragover', 'dragenter'].forEach(evt =>
  dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.add('dragover'); })
);
['dragleave', 'drop'].forEach(evt =>
  dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); })
);
dropZone.addEventListener('drop', (e) => {
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
imageInput.addEventListener('change', () => {
  if (imageInput.files.length) handleFile(imageInput.files[0]);
});

function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    showError('Please select an image file (JPEG, PNG, WEBP).');
    return;
  }
  selectedFile = file;
  hideError();

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    previewImage.hidden = false;
    dropZoneContent.hidden = true;
    predictBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.hidden = false;
}
function hideError() {
  errorMsg.hidden = true;
}

predictBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  hideError();
  uploadCard.hidden = true;
  loadingState.hidden = false;
  resultsSection.hidden = true;

  let msgIndex = 0;
  loadingText.textContent = LOADING_MESSAGES[0];
  const loadingInterval = setInterval(() => {
    msgIndex = (msgIndex + 1) % LOADING_MESSAGES.length;
    loadingText.textContent = LOADING_MESSAGES[msgIndex];
  }, 1400);

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const response = await fetch('/predict', { method: 'POST', body: formData });
    const data = await response.json();

    clearInterval(loadingInterval);
    loadingState.hidden = true;

    if (!response.ok) {
      uploadCard.hidden = false;
      showError(data.error || 'Something went wrong. Please try again.');
      return;
    }

    renderResults(data);
    resultsSection.hidden = false;
  } catch (err) {
    clearInterval(loadingInterval);
    loadingState.hidden = true;
    uploadCard.hidden = false;
    showError('Could not reach the server. Is app.py running?');
  }
});

function renderResults(data) {
  document.getElementById('res-plant').textContent = data.plant;

  const statusPill = document.getElementById('res-status');
  statusPill.textContent = data.health_status;
  statusPill.className = 'status-pill ' + (data.health_status === 'Healthy' ? 'healthy' : 'diseased');

  setGauge('plant-gauge', 'plant-confidence-value', data.plant_confidence);
  setGauge('disease-gauge', 'disease-confidence-value', data.disease_confidence);

  document.getElementById('uncertainty-note').hidden = !data.plant_identification_uncertain;

  document.getElementById('res-disease').textContent = data.disease;
  document.getElementById('res-symptoms').textContent = data.symptoms;
  document.getElementById('res-causes').textContent = data.causes;

  const topkList = document.getElementById('res-topk');
  topkList.innerHTML = '';
  data.top_k_predictions.forEach((item) => {
    const li = document.createElement('li');
    li.innerHTML = `<span>${item.disease}</span><span>${(item.confidence * 100).toFixed(1)}%</span>`;
    topkList.appendChild(li);
  });

  document.getElementById('gradcam-image').src = data.gradcam_image;
  document.getElementById('res-rationale').textContent = data.explanation;

  document.getElementById('res-organic').textContent = data.organic_treatment;
  document.getElementById('res-chemical').textContent = data.chemical_treatment;
  document.getElementById('res-fertilizer').textContent = data.fertilizer;
  document.getElementById('res-watering').textContent = data.watering;
  document.getElementById('res-prevention').textContent = data.prevention;
  document.getElementById('res-future').textContent = data.future_care;
}

function setGauge(gaugeId, valueId, confidence) {
  const pct = Math.round(confidence * 100);
  document.getElementById(gaugeId).style.width = pct + '%';
  document.getElementById(valueId).textContent = pct + '%';
}

resetBtn.addEventListener('click', () => {
  selectedFile = null;
  imageInput.value = '';
  previewImage.hidden = true;
  dropZoneContent.hidden = false;
  predictBtn.disabled = true;
  resultsSection.hidden = true;
  uploadCard.hidden = false;
  hideError();
});
