'use strict';

/* Pixel Party XL — browser client.
   Generation is a job: POST once, then poll. The polling loop is the only
   thing that writes progress UI, so there is a single source of truth for
   what the page is showing. */

const $ = (selector) => document.querySelector(selector);
const STORAGE_KEY = 'pixel-party-xl:settings';
const POLL_MS = 700;

const PRESETS = [
  'cute dragon',
  'knight with a big sword',
  'health potion bottle',
  'forest tileset',
  'treasure chest',
  'wizard character sprite',
  'small spaceship',
  'castle on a hill',
];

const state = {
  config: null,
  ready: false,
  jobId: null,
  pollTimer: null,
  pollFailures: 0,
  count: 1,
  initImage: null,
  submitting: false,
};

/* ------------------------------------------------------------------ utils */

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(describeError(body) || `HTTP ${response.status}`);
  }
  return body;
}

/** FastAPI reports validation errors as a list and everything else as a
    string, so both shapes have to be unwrapped before display. */
function describeError(body) {
  if (!body || !body.detail) return null;
  if (typeof body.detail === 'string') return body.detail;
  if (Array.isArray(body.detail)) {
    return body.detail.map((item) => item.msg || String(item)).join(' · ');
  }
  return null;
}

let toastTimer = null;
function toast(message, kind = 'error') {
  const element = $('#toast');
  element.textContent = message;
  element.dataset.kind = kind;
  element.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { element.hidden = true; }, 5200);
}

function formatSeconds(value) {
  if (value === null || value === undefined) return '';
  if (value < 60) return `${Math.ceil(value)}sn`;
  return `${Math.floor(value / 60)}dk ${Math.ceil(value % 60)}sn`;
}

/* ------------------------------------------------------------ setup / boot */

async function boot() {
  wireStaticControls();
  try {
    state.config = await fetchJSON('/api/config');
  } catch (error) {
    toast(`Ayarlar alınamadı: ${error.message}`);
    return;
  }
  buildControls(state.config);
  restoreSettings();
  syncDerivedLabels();
  pollHealth();
}

function buildControls(config) {
  const sizeSelect = $('#size');
  sizeSelect.innerHTML = '';
  config.sizes.forEach(({ width, height }) => {
    const option = document.createElement('option');
    option.value = `${width}x${height}`;
    // Just the numbers: they already convey the shape, and anything longer
    // gets clipped in a half-width select.
    option.textContent = `${width}×${height}`;
    sizeSelect.append(option);
  });
  sizeSelect.value = `${config.defaults.width}x${config.defaults.height}`;

  const schedulerSelect = $('#scheduler');
  schedulerSelect.innerHTML = '';
  config.schedulers.forEach(({ id, label }) => {
    const option = document.createElement('option');
    option.value = id;
    option.textContent = label;
    schedulerSelect.append(option);
  });
  schedulerSelect.value = config.defaults.scheduler;

  const factorSelect = $('#pixel-factor');
  factorSelect.innerHTML = '';
  config.pixel_factors.forEach((factor) => {
    const option = document.createElement('option');
    option.value = String(factor);
    factorSelect.append(option);
  });
  factorSelect.value = String(config.defaults.pixel_factor);

  const countGroup = $('#count');
  countGroup.innerHTML = '';
  for (let n = 1; n <= config.limits.max_images; n += 1) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = String(n);
    button.setAttribute('aria-pressed', String(n === state.count));
    button.addEventListener('click', () => {
      state.count = n;
      [...countGroup.children].forEach((child, index) => {
        child.setAttribute('aria-pressed', String(index + 1 === n));
      });
      saveSettings();
    });
    countGroup.append(button);
  }

  $('#negative').value = config.defaults.negative_prompt;
  $('#steps').value = config.defaults.steps;
  $('#guidance').value = config.defaults.guidance;
  $('#strength').value = config.defaults.strength;

  const presets = $('#presets');
  PRESETS.forEach((text) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'chip';
    chip.textContent = text;
    chip.addEventListener('click', () => {
      $('#prompt').value = text;
      syncDerivedLabels();
      saveSettings();
    });
    presets.append(chip);
  });
}

function wireStaticControls() {
  $('#steps').addEventListener('input', syncDerivedLabels);
  $('#guidance').addEventListener('input', syncDerivedLabels);
  $('#strength').addEventListener('input', syncDerivedLabels);
  $('#size').addEventListener('change', () => { syncDerivedLabels(); saveSettings(); });
  $('#pixel-factor').addEventListener('change', saveSettings);
  $('#palette').addEventListener('change', saveSettings);
  $('#scheduler').addEventListener('change', saveSettings);
  $('#negative').addEventListener('change', saveSettings);
  $('#append-style').addEventListener('change', saveSettings);
  $('#prompt').addEventListener('input', () => { syncDerivedLabels(); saveSettings(); });

  $('#generate').addEventListener('click', startGeneration);
  $('#cancel').addEventListener('click', cancelGeneration);

  $('#seed-dice').addEventListener('click', () => {
    $('#seed').value = Math.floor(Math.random() * 2147483647);
  });
  $('#seed-clear').addEventListener('click', () => { $('#seed').value = ''; });

  wireInitImage();

  // Ctrl/Cmd+Enter from anywhere in the form fires a run.
  document.addEventListener('keydown', (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
      event.preventDefault();
      startGeneration();
    }
  });
}

/** Keeps every readout that is derived from another control in sync. */
function syncDerivedLabels() {
  $('#steps-out').textContent = $('#steps').value;
  $('#guidance-out').textContent = Number($('#guidance').value).toFixed(1);
  $('#strength-out').textContent = Number($('#strength').value).toFixed(2);

  const counter = $('#prompt-counter');
  const length = $('#prompt').value.length;
  counter.textContent = String(length);
  counter.classList.toggle('over', length > 300);

  // Show what each downscale factor actually produces for the chosen canvas,
  // because "8×" means nothing without the resulting sprite size next to it.
  const [width, height] = $('#size').value.split('x').map(Number);
  [...$('#pixel-factor').options].forEach((option) => {
    const factor = Number(option.value);
    option.textContent =
      factor === 1
        ? '1× · küçültme yok'
        : `${factor}× → ${Math.floor(width / factor)}×${Math.floor(height / factor)}`;
  });
}

/* ------------------------------------------------------------ persistence */

function saveSettings() {
  const settings = {
    prompt: $('#prompt').value,
    negative: $('#negative').value,
    appendStyle: $('#append-style').checked,
    steps: $('#steps').value,
    guidance: $('#guidance').value,
    size: $('#size').value,
    scheduler: $('#scheduler').value,
    pixelFactor: $('#pixel-factor').value,
    palette: $('#palette').value,
    count: state.count,
  };
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    /* private mode or a full quota — settings just will not persist. */
  }
}

function restoreSettings() {
  let settings;
  try {
    settings = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
  } catch {
    settings = null;
  }
  if (!settings) return;

  const setIfPresent = (selector, value) => {
    if (value !== undefined && value !== null && value !== '') $(selector).value = value;
  };
  setIfPresent('#prompt', settings.prompt);
  setIfPresent('#negative', settings.negative);
  setIfPresent('#steps', settings.steps);
  setIfPresent('#guidance', settings.guidance);
  setIfPresent('#palette', settings.palette);
  if (settings.appendStyle !== undefined) $('#append-style').checked = settings.appendStyle;

  // Only restore list-backed values that still exist in this build's config.
  const restoreOption = (selector, value) => {
    const select = $(selector);
    if (value && [...select.options].some((option) => option.value === value)) {
      select.value = value;
    }
  };
  restoreOption('#size', settings.size);
  restoreOption('#scheduler', settings.scheduler);
  restoreOption('#pixel-factor', settings.pixelFactor);

  if (settings.count) {
    state.count = settings.count;
    [...$('#count').children].forEach((child, index) => {
      child.setAttribute('aria-pressed', String(index + 1 === settings.count));
    });
  }
}

/* ------------------------------------------------------------------ health */

async function pollHealth() {
  const status = $('#status');
  const text = status.querySelector('.status-text');
  try {
    const health = await fetchJSON('/api/health');
    state.ready = Boolean(health.ready);

    if (health.status === 'error') {
      status.dataset.state = 'error';
      text.textContent = 'Model yüklenemedi';
      toast(health.error || 'Model yüklenemedi.');
      setGenerateEnabled(false);
      return;
    }
    if (health.ready) {
      status.dataset.state = 'ready';
      const gpu = health.gpu ? health.gpu.replace('NVIDIA ', '') : health.device;
      text.textContent = `Hazır · ${gpu}`;
      setGenerateEnabled(true);
      return;
    }
    status.dataset.state = 'loading';
    text.textContent = 'Model yükleniyor…';
    setGenerateEnabled(false);
  } catch {
    status.dataset.state = 'loading';
    text.textContent = 'Sunucu bekleniyor…';
    setGenerateEnabled(false);
  }
  setTimeout(pollHealth, 2500);
}

function setGenerateEnabled(enabled) {
  $('#generate').disabled = !enabled;
}

/* -------------------------------------------------------------- generation */

function collectRequest() {
  const [width, height] = $('#size').value.split('x').map(Number);
  const seedRaw = $('#seed').value.trim();
  const payload = {
    prompt: $('#prompt').value.trim(),
    negative_prompt: $('#negative').value,
    append_style: $('#append-style').checked,
    steps: Number($('#steps').value),
    guidance: Number($('#guidance').value),
    width,
    height,
    num_images: state.count,
    seed: seedRaw === '' ? null : Number(seedRaw),
    scheduler: $('#scheduler').value,
    pixel_factor: Number($('#pixel-factor').value),
    palette_colors: Number($('#palette').value),
  };
  if (state.initImage) {
    payload.init_image = state.initImage;
    payload.strength = Number($('#strength').value);
  }
  return payload;
}

async function startGeneration() {
  // jobId alone is not enough: the Ctrl+Enter shortcut can fire again while
  // the POST is still in flight, before there is any job id to guard on.
  if (state.jobId || state.submitting) return;
  if (!state.ready) {
    toast('Model hâlâ yükleniyor, birkaç saniye sonra dene.');
    return;
  }
  const payload = collectRequest();
  if (!payload.prompt) {
    toast('Önce bir prompt yaz.');
    $('#prompt').focus();
    return;
  }

  state.submitting = true;
  setBusy(true);
  try {
    const { job_id: jobId } = await fetchJSON('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    state.jobId = jobId;
    state.pollFailures = 0;
    pollJob();
  } catch (error) {
    setBusy(false);
    toast(error.message);
  } finally {
    state.submitting = false;
  }
}

async function cancelGeneration() {
  if (!state.jobId) return;
  $('#cancel').disabled = true;
  try {
    await fetch(`/api/job/${state.jobId}/cancel`, { method: 'POST' });
    $('#progress-label').textContent = 'İptal ediliyor…';
  } catch {
    toast('İptal isteği gönderilemedi.');
  }
}

async function pollJob() {
  if (!state.jobId) return;
  let job;
  try {
    job = await fetchJSON(`/api/job/${state.jobId}`);
    state.pollFailures = 0;
  } catch (error) {
    // A dropped poll is usually a hiccup, not a dead job — give it a few
    // tries before tearing the UI down.
    state.pollFailures += 1;
    if (state.pollFailures > 6) {
      finishJob(null, `Durum alınamadı: ${error.message}`);
      return;
    }
    state.pollTimer = setTimeout(pollJob, POLL_MS * 2);
    return;
  }

  renderProgress(job);

  if (job.status === 'done' || job.status === 'cancelled' || job.status === 'error') {
    finishJob(job, null);
    return;
  }
  state.pollTimer = setTimeout(pollJob, POLL_MS);
}

function renderProgress(job) {
  const bar = $('#progress-bar');
  const label = $('#progress-label');
  const meta = $('#progress-meta');

  if (job.status === 'queued') {
    label.textContent = job.queue_position
      ? `Kuyrukta · ${job.queue_position}. sıra`
      : 'Kuyrukta…';
    meta.textContent = '';
    bar.style.width = '4%';
    return;
  }

  label.textContent = 'Üretiliyor…';
  bar.style.width = `${Math.max(4, job.progress * 100).toFixed(1)}%`;
  const parts = [`${job.step}/${job.total_steps} adım`];
  if (job.eta_seconds !== null && job.eta_seconds !== undefined && job.eta_seconds > 0) {
    parts.push(`~${formatSeconds(job.eta_seconds)}`);
  }
  meta.textContent = parts.join(' · ');
}

function finishJob(job, errorMessage) {
  clearTimeout(state.pollTimer);
  state.jobId = null;
  setBusy(false);

  if (errorMessage) {
    toast(errorMessage);
    return;
  }
  if (job.images && job.images.length) {
    renderResults(job);
  }
  if (job.status === 'error') {
    toast(job.error || 'Üretim başarısız oldu.');
  } else if (job.status === 'cancelled') {
    toast('İptal edildi.', 'info');
  } else if (job.images && job.images.length) {
    toast(`${job.images.length} görsel · ${formatSeconds(job.elapsed)}`, 'info');
  }
}

function setBusy(busy) {
  $('#generate').hidden = busy;
  $('#cancel').hidden = !busy;
  $('#cancel').disabled = false;
  $('#progress').hidden = !busy;
  if (busy) {
    $('#progress-bar').style.width = '2%';
    $('#progress-label').textContent = 'Gönderiliyor…';
    $('#progress-meta').textContent = '';
  }
}

/* ---------------------------------------------------------------- results */

function renderResults(job) {
  $('#empty').hidden = true;
  const gallery = $('#gallery');
  const template = $('#card-template');

  // Newest first, but keep this job's own images in generation order.
  [...job.images].reverse().forEach((image) => {
    const card = template.content.cloneNode(true).firstElementChild;
    const img = card.querySelector('img');
    const toggle = card.querySelector('.view-toggle');
    const dims = card.querySelector('.dims');
    const seedChip = card.querySelector('.seed-chip');

    const scale = Math.max(1, Math.round(image.width / image.pixel_width));

    img.src = image.pixel_url;
    img.alt = `${job.id} · seed ${image.seed}`;
    dims.textContent = `${image.pixel_width}×${image.pixel_height}`;
    seedChip.textContent = `#${image.seed}`;

    toggle.addEventListener('click', () => {
      const showingPixel = toggle.dataset.view === 'pixel';
      toggle.dataset.view = showingPixel ? 'raw' : 'pixel';
      img.src = showingPixel ? image.raw_url : image.pixel_url;
      dims.textContent = showingPixel
        ? `${image.width}×${image.height}`
        : `${image.pixel_width}×${image.pixel_height}`;
      // The 1024px render is being shown scaled *down* in the card, where
      // nearest-neighbour would alias it badly. Only the true sprite wants
      // hard edges.
      img.style.imageRendering = showingPixel ? 'auto' : 'pixelated';
    });

    seedChip.addEventListener('click', () => {
      $('#seed').value = image.seed;
      toast(`Seed ${image.seed} kilitlendi.`, 'info');
    });

    card.querySelector('[data-role="dl-pixel"]').href = `${image.pixel_url}?download=1`;
    card.querySelector('[data-role="dl-raw"]').href = `${image.raw_url}?download=1`;

    const scaled = card.querySelector('[data-role="dl-scaled"]');
    if (scale > 1) {
      scaled.href = `${image.pixel_url}?scale=${scale}&download=1`;
      scaled.textContent = `PNG ${scale}×`;
    } else {
      scaled.remove();
      card.querySelector('.card-actions').style.gridTemplateColumns = 'repeat(3, 1fr)';
    }

    card.querySelector('[data-role="use-init"]').addEventListener('click', () => {
      useAsInit(image.pixel_url);
    });

    gallery.prepend(card);
  });
}

/* ------------------------------------------------------------- init image */

function wireInitImage() {
  const dropzone = $('#dropzone');
  const fileInput = $('#init-file');

  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files[0]) acceptInitFile(fileInput.files[0]);
  });

  ['dragenter', 'dragover'].forEach((type) => {
    dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dropzone.classList.add('drag');
    });
  });
  ['dragleave', 'drop'].forEach((type) => {
    dropzone.addEventListener(type, () => dropzone.classList.remove('drag'));
  });
  dropzone.addEventListener('drop', (event) => {
    event.preventDefault();
    const file = event.dataTransfer?.files?.[0];
    if (file) acceptInitFile(file);
  });

  $('#init-clear').addEventListener('click', (event) => {
    event.stopPropagation();
    clearInitImage();
  });

  document.addEventListener('paste', (event) => {
    const item = [...(event.clipboardData?.items || [])].find((entry) =>
      entry.type.startsWith('image/'),
    );
    if (!item) return;
    const file = item.getAsFile();
    if (file) {
      $('#init-group').open = true;
      acceptInitFile(file);
    }
  });
}

async function acceptInitFile(file) {
  if (!file.type.startsWith('image/')) {
    toast('Bu bir görsel dosyası değil.');
    return;
  }
  try {
    setInitImage(await normalizeImage(file));
  } catch (error) {
    toast(`Görsel okunamadı: ${error.message}`);
  }
}

/** Read a file as a data URL, shrinking it first only if it is genuinely big.
    Small files are passed through byte-for-byte: re-encoding a sprite through
    a canvas would resample away the hard pixel edges that matter here. */
function normalizeImage(file) {
  const PASSTHROUGH_LIMIT = 4 * 1024 * 1024;
  const MAX_EDGE = 1536;

  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error('dosya okunamadı'));
    reader.onload = () => {
      const dataUrl = reader.result;
      if (file.size <= PASSTHROUGH_LIMIT) {
        resolve(dataUrl);
        return;
      }
      const image = new Image();
      image.onerror = () => reject(new Error('görsel çözülemedi'));
      image.onload = () => {
        const ratio = Math.min(1, MAX_EDGE / Math.max(image.width, image.height));
        const canvas = document.createElement('canvas');
        canvas.width = Math.round(image.width * ratio);
        canvas.height = Math.round(image.height * ratio);
        const context = canvas.getContext('2d');
        context.imageSmoothingQuality = 'high';
        context.drawImage(image, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL('image/jpeg', 0.92));
      };
      image.src = dataUrl;
    };
    reader.readAsDataURL(file);
  });
}

function setInitImage(dataUrl) {
  state.initImage = dataUrl;
  $('#init-preview').src = dataUrl;
  $('#dropzone').querySelector('.dropzone-empty').hidden = true;
  $('#dropzone').querySelector('.dropzone-preview').hidden = false;
  const tag = $('#init-tag');
  tag.textContent = 'açık';
  tag.classList.add('on');
}

function clearInitImage() {
  state.initImage = null;
  $('#init-file').value = '';
  $('#init-preview').removeAttribute('src');
  $('#dropzone').querySelector('.dropzone-empty').hidden = false;
  $('#dropzone').querySelector('.dropzone-preview').hidden = true;
  const tag = $('#init-tag');
  tag.textContent = 'kapalı';
  tag.classList.remove('on');
}

async function useAsInit(url) {
  try {
    const blob = await (await fetch(url)).blob();
    const dataUrl = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error('okunamadı'));
      reader.onload = () => resolve(reader.result);
      reader.readAsDataURL(blob);
    });
    setInitImage(dataUrl);
    $('#init-group').open = true;
    $('#init-group').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    toast('Başlangıç görseli olarak ayarlandı.', 'info');
  } catch {
    toast('Görsel init olarak ayarlanamadı.');
  }
}

boot();
