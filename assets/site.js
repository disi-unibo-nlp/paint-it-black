const navToggle = document.querySelector('.nav-toggle');
const navLinks = document.querySelector('.nav-links');

navToggle?.addEventListener('click', () => {
  const open = navToggle.getAttribute('aria-expanded') === 'true';
  navToggle.setAttribute('aria-expanded', String(!open));
  navLinks?.classList.toggle('open', !open);
});

navLinks?.querySelectorAll('a').forEach((link) => {
  link.addEventListener('click', () => {
    navToggle?.setAttribute('aria-expanded', 'false');
    navLinks.classList.remove('open');
  });
});

const redactionStage = document.querySelector('.redaction-stage');
document.querySelectorAll('[data-mode-button]').forEach((button) => {
  button.addEventListener('click', () => {
    const mode = button.dataset.modeButton;
    if (!mode || !redactionStage) return;
    redactionStage.dataset.mode = mode;
    document.querySelectorAll('[data-mode-button]').forEach((item) => item.classList.remove('active'));
    button.classList.add('active');
  });
});

const metrics = {
  text: {
    description: 'Exact PHI span extraction, ignoring bounding-box placement.',
    values: [
      ['Gemma-4-31B', 73.7],
      ['Qwen3.5-27B', 72.5],
      ['Qwen3.5-122B-A10B', 71.4],
      ['Gemma-4-26B-A4B', 69.0],
      ['Qwen3.5-9B', 64.5],
      ['Qwen3.6-35B-A3B', 64.3],
      ['InternVL3.5-14B', 50.1],
      ['MedGemma-27B', 2.3]
    ]
  },
  spatial: {
    description: 'Bounding-box localization, averaged across IoU thresholds and ignoring text transcription.',
    values: [
      ['Gemma-4-31B', 57.9],
      ['Gemma-4-26B-A4B', 54.8],
      ['Qwen3.5-27B', 40.3],
      ['Qwen3.6-35B-A3B', 17.5],
      ['Qwen3.5-122B-A10B', 14.5],
      ['Qwen3.5-9B', 2.1],
      ['InternVL3.5-14B', 0.0],
      ['MedGemma-27B', 0.0]
    ]
  },
  pass: {
    description: 'Privacy-critical joint correctness: exact text and IoU > 0.5 must both hold.',
    values: [
      ['Gemma-4-31B', 61.6],
      ['Gemma-4-26B-A4B', 57.6],
      ['Qwen3.5-27B', 37.7],
      ['Qwen3.6-35B-A3B', 11.9],
      ['Qwen3.5-122B-A10B', 10.2],
      ['Qwen3.5-9B', 0.9],
      ['InternVL3.5-14B', 0.0],
      ['MedGemma-27B', 0.0]
    ]
  }
};

const modelChart = document.querySelector('#model-chart');
const metricDescription = document.querySelector('#metric-description');

function renderMetric(metricName) {
  const metric = metrics[metricName];
  if (!metric || !modelChart || !metricDescription) return;
  metricDescription.textContent = metric.description;
  modelChart.replaceChildren();

  metric.values.forEach(([name, value]) => {
    const row = document.createElement('div');
    row.className = 'model-row';

    const label = document.createElement('span');
    label.className = 'model-name';
    label.textContent = name;

    const track = document.createElement('div');
    track.className = 'model-track';
    const bar = document.createElement('div');
    bar.className = 'model-bar';
    bar.style.setProperty('--value', `${value}%`);
    track.appendChild(bar);

    const score = document.createElement('span');
    score.className = 'model-value';
    score.textContent = value.toFixed(1);

    row.append(label, track, score);
    modelChart.appendChild(row);
  });
}

document.querySelectorAll('.metric-tabs button').forEach((button) => {
  button.addEventListener('click', () => {
    const metric = button.dataset.metric;
    if (!metric) return;
    document.querySelectorAll('.metric-tabs button').forEach((item) => item.setAttribute('aria-selected', 'false'));
    button.setAttribute('aria-selected', 'true');
    renderMetric(metric);
  });
});

renderMetric('text');

const copyButton = document.querySelector('.copy-button');
copyButton?.addEventListener('click', async () => {
  const code = document.querySelector('#bibtex-code')?.textContent;
  if (!code) return;
  try {
    await navigator.clipboard.writeText(code);
    copyButton.textContent = 'Copied';
    setTimeout(() => { copyButton.textContent = 'Copy'; }, 1600);
  } catch {
    copyButton.textContent = 'Select text';
  }
});

const lightbox = document.querySelector('#figure-lightbox');
const lightboxImage = document.querySelector('#lightbox-image');
const lightboxCaption = document.querySelector('#lightbox-caption');
const lightboxClose = document.querySelector('.lightbox-close');

document.querySelectorAll('.zoom-trigger').forEach((trigger) => {
  trigger.addEventListener('click', () => {
    const source = trigger.querySelector('img');
    if (!source || !lightbox || !lightboxImage || !lightboxCaption) return;
    lightboxImage.src = source.currentSrc || source.src;
    lightboxImage.alt = source.alt;
    lightboxCaption.textContent = source.alt;
    lightbox.showModal();
    document.body.classList.add('lightbox-open');
  });
});

lightboxClose?.addEventListener('click', () => lightbox?.close());
lightbox?.addEventListener('click', (event) => {
  if (event.target === lightbox) lightbox.close();
});
lightbox?.addEventListener('close', () => {
  document.body.classList.remove('lightbox-open');
  if (lightboxImage) lightboxImage.src = '';
});

const soundtrack = document.querySelector('[data-soundtrack]');
const soundtrackTrigger = soundtrack?.querySelector('.soundtrack-trigger');
const soundtrackPlayer = soundtrack?.querySelector('.soundtrack-player');
const soundtrackAction = soundtrack?.querySelector('[data-soundtrack-action]');
const spotifyEmbed = document.querySelector('#spotify-embed');
let spotifyController = null;
let playWhenReady = false;

function updateSoundtrackState(playing) {
  if (!soundtrack || !soundtrackTrigger || !soundtrackAction) return;
  soundtrack.dataset.playing = String(playing);
  soundtrackTrigger.setAttribute('aria-pressed', String(playing));
  soundtrackAction.textContent = playing ? 'Ⅱ' : '▶';
}

window.onSpotifyIframeApiReady = (IFrameAPI) => {
  if (!spotifyEmbed) return;
  IFrameAPI.createController(spotifyEmbed, {
    width: '100%',
    height: 80,
    uri: 'spotify:track:3qWi17QFSrLgj0g87zGwc6'
  }, (controller) => {
    spotifyController = controller;
    controller.addListener('playback_update', (event) => {
      updateSoundtrackState(!event.data.isPaused);
    });
    if (playWhenReady) {
      controller.play();
      playWhenReady = false;
    }
  });
};

if (soundtrack && spotifyEmbed) {
  const spotifyScript = document.createElement('script');
  spotifyScript.src = 'https://open.spotify.com/embed/iframe-api/v1';
  spotifyScript.async = true;
  document.body.appendChild(spotifyScript);
}

soundtrackTrigger?.addEventListener('click', () => {
  soundtrack?.classList.add('expanded');
  soundtrackTrigger.setAttribute('aria-expanded', 'true');
  soundtrackPlayer?.setAttribute('aria-hidden', 'false');
  if (spotifyController) {
    spotifyController.togglePlay();
  } else {
    playWhenReady = true;
  }
});

const revealItems = document.querySelectorAll('.reveal');
if ('IntersectionObserver' in window && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.08 });
  revealItems.forEach((item) => observer.observe(item));
} else {
  revealItems.forEach((item) => item.classList.add('visible'));
}
