// ECObage - определение материала идёт через сервер (app.py, обученная модель из training/),
// а COCO-SSD (человек) и Tesseract OCR (мем "6 7") работают прямо в браузере

const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const MIN_ANALYZE_MS = 1400; // чтобы анимация анализа была заметна даже при быстрой модели

const CATEGORIES = {
  plastic: {
    label: 'Пластик',
    emoji: '🟡',
    color: '#f2b705',
    bg: '#fef7de',
    explain: 'Это пластик. Он разлагается очень долго, но перерабатывается.',
    sorting: 'Выбросьте в контейнер для пластика.'
  },
  glass: {
    label: 'Стекло',
    emoji: '🟢',
    color: '#2e9e5b',
    bg: '#e6f6ec',
    explain: 'Это стекло. Его можно перерабатывать много раз без потери качества.',
    sorting: 'Снимите крышку и выбросьте в контейнер для стекла.'
  },
  organic: {
    label: 'Органика',
    emoji: '🟤',
    color: '#8a5a34',
    bg: '#f3e9e0',
    explain: 'Это органические отходы. Они разлагаются естественным путём.',
    sorting: 'Выбросьте в контейнер для органических отходов или в компост.'
  },
  paper: {
    label: 'Бумага',
    emoji: '🔵',
    color: '#2f7fe0',
    bg: '#e8f1fc',
    explain: 'Это бумага или картон. Такой материал перерабатывается.',
    sorting: 'Сложите и выбросьте в контейнер для бумаги.'
  },
  metal: {
    label: 'Металл',
    emoji: '⚪',
    color: '#7d868c',
    bg: '#eef1f2',
    explain: 'Это металл. Он хорошо поддаётся переплавке и переработке.',
    sorting: 'Выбросьте в контейнер для металла.'
  }
};

// Классы собственной модели (training/train.py), обученной на датасете из archive (1).zip,
// сопоставлены с категориями отходов интерфейса. Apple - отдельная проверка через ImageNet
// (см. app.py) на случай яблок, которых в датасете мусора нет вообще.
const CLASS_MAP = {
  Bouteille_plastique: { cat: 'plastic', ru: 'Пластиковая бутылка' },
  Brique_en_carton: { cat: 'paper', ru: 'Картонная упаковка (тетрапак)' },
  Emballage_metallique: { cat: 'metal', ru: 'Металлическая упаковка' },
  Ordure_menagere: { cat: 'organic', ru: 'Бытовые отходы' },
  Papier_Carton: { cat: 'paper', ru: 'Бумага / картон' },
  Verre: { cat: 'glass', ru: 'Стеклянная тара' },
  Apple: { cat: 'organic', ru: 'Яблоко' }
};

const MATERIAL_MIN_CONFIDENCE = 0.35;

const PERSON_MIN_SCORE = 0.5;
const MEME_MIN_CONFIDENCE = 20; // Tesseract OCR confidence, шкала 0-100 (мультяшный шрифт даёт низкую уверенность)

let cocoSsdModel = null;
let cocoSsdLoadingPromise = null;
let tesseractWorkerPromise = null;

function getTesseractWorker() {
  if (!tesseractWorkerPromise) {
    tesseractWorkerPromise = Tesseract.createWorker('eng');
  }
  return tesseractWorkerPromise;
}

// Узнаёт вирусный мем "6 7" (число 67 крупным текстом + раскрытые ладони на фото).
// Возвращает { matched, confidence } — matched, если распознанный текст — это, по сути, "67".
async function detectSixSevenMeme(imgEl) {
  try {
    const worker = await getTesseractWorker();
    const { data } = await worker.recognize(imgEl);
    const digitsOnly = (data.text || '').replace(/[^0-9]/g, '');
    // мультяшный шрифт на картинке иногда даёт лишний символ рядом с "67",
    // поэтому проверяем не строгое равенство, а короткую строку с "67" внутри
    if (digitsOnly.includes('67') && digitsOnly.length <= 4) {
      return { matched: true, confidence: data.confidence };
    }
    return { matched: false, confidence: 0 };
  } catch (err) {
    console.warn('OCR-проверка мема "6 7" не удалась:', err);
    return { matched: false, confidence: 0 };
  }
}

const els = {
  screenUpload: document.getElementById('screen-upload'),
  screenAnalyze: document.getElementById('screen-analyze'),
  screenResult: document.getElementById('screen-result'),
  dropzone: document.getElementById('dropzone'),
  fileInput: document.getElementById('fileInput'),
  uploadBtn: document.getElementById('uploadBtn'),
  previewImg: document.getElementById('previewImg'),
  resultImg: document.getElementById('resultImg'),
  resultCard: document.getElementById('resultCard'),
  analyzeText: document.getElementById('analyzeText'),
  resetBtn: document.getElementById('resetBtn'),
  toast: document.getElementById('toast')
};

function showScreen(name) {
  [els.screenUpload, els.screenAnalyze, els.screenResult].forEach(s => s.classList.remove('active'));
  ({ upload: els.screenUpload, analyze: els.screenAnalyze, result: els.screenResult })[name].classList.add('active');
}

function showToast(message) {
  els.toast.textContent = message;
  els.toast.classList.add('show');
  setTimeout(() => els.toast.classList.remove('show'), 3200);
}

function getCocoSsdModel() {
  if (cocoSsdModel) return Promise.resolve(cocoSsdModel);
  if (!cocoSsdLoadingPromise) {
    cocoSsdLoadingPromise = cocoSsd.load().then(m => {
      cocoSsdModel = m;
      return m;
    });
  }
  return cocoSsdLoadingPromise;
}

// Начинаем загрузку COCO-SSD сразу при открытии страницы, чтобы первый анализ был быстрее.
// COCO-SSD нужен, чтобы понять, есть ли на фото человек (у ImageNet-моделей почти нет
// класса «человек», а COCO-SSD обучен именно на классе "person").
getCocoSsdModel().catch(err => console.error('Не удалось загрузить модель COCO-SSD:', err));

// Превращает картинку/canvas в Blob, чтобы отправить на сервер.
function elementToBlob(el) {
  return new Promise(resolve => {
    if (el.toBlob) {
      el.toBlob(resolve, 'image/jpeg', 0.9);
      return;
    }
    const canvas = document.createElement('canvas');
    canvas.width = el.naturalWidth;
    canvas.height = el.naturalHeight;
    canvas.getContext('2d').drawImage(el, 0, 0);
    canvas.toBlob(resolve, 'image/jpeg', 0.9);
  });
}

// Определение материала теперь делает сервер (app.py) - там же, где обученная модель.
async function classifyMaterial(imgEl) {
  const blob = await elementToBlob(imgEl);
  const formData = new FormData();
  formData.append('image', blob, 'photo.jpg');

  const response = await fetch('/analyze', { method: 'POST', body: formData });
  const data = await response.json();
  return { className: data.class_name, probability: data.probability };
}

async function classifyImage(imgEl) {
  const detector = await getCocoSsdModel();

  const [memeResult, detections, material] = await Promise.all([
    detectSixSevenMeme(imgEl),
    detector.detect(imgEl),
    classifyMaterial(imgEl)
  ]);

  // Приоритет: мем > человек > материал отхода.
  // Мем "6 7" (и подобные картинки без мусора) - это не отходы, поэтому здесь
  // не подбираем категорию, а сразу говорим, что объект не найден.
  if (memeResult.matched && memeResult.confidence >= MEME_MIN_CONFIDENCE) {
    return { matched: false, confidence: 0, rawLabel: '67-meme' };
  }

  const person = detections
    .filter(d => d.class === 'person' && d.score >= PERSON_MIN_SCORE)
    .sort((a, b) => b.score - a.score)[0];

  if (person) {
    return {
      matched: true,
      isPerson: true,
      rule: { cat: 'organic', ru: 'Человек' },
      confidence: person.score,
      rawLabel: 'person'
    };
  }

  // Яблоко - отдельный случай: наш датасет мусора яблок не содержит, поэтому сервер
  // проверяет его через общий ImageNet и не подчиняется общему порогу уверенности.
  if (material.className === 'Apple') {
    return {
      matched: true,
      rule: CLASS_MAP.Apple,
      confidence: material.probability,
      rawLabel: material.className
    };
  }

  const rule = CLASS_MAP[material.className];
  if (rule && material.probability >= MATERIAL_MIN_CONFIDENCE) {
    return {
      matched: true,
      rule,
      confidence: material.probability,
      rawLabel: material.className
    };
  }

  return {
    matched: false,
    confidence: material.probability,
    rawLabel: material.className
  };
}

function renderResult(classification) {
  const confidencePct = Math.round(classification.confidence * 100);

  if (!classification.matched) {
    els.resultCard.className = 'result-card uncertain';
    els.resultCard.innerHTML = `
      <div class="uncertain-icon">🤔</div>
      <p class="uncertain-text">Не удалось определить объект. Пожалуйста, загрузите фотографию отходов.</p>
    `;
    return;
  }

  const cat = CATEGORIES[classification.rule.cat];
  const explain = classification.isPerson
    ? 'Человек не является отходом.'
    : cat.explain;
  const sorting = classification.isPerson
    ? 'Категория сортировки здесь не применяется.'
    : cat.sorting;

  els.resultCard.className = 'result-card';
  els.resultCard.innerHTML = `
    <div class="cat-badge" style="background:${cat.bg}; color:${cat.color};">
      <span class="cat-icon">${cat.emoji}</span> ${cat.label}
    </div>
    <h2 class="item-name">${classification.rule.ru}</h2>
    <div class="confidence-row">
      <span class="confidence-label">Вероятность</span>
      <div class="confidence-bar"><div class="confidence-fill" style="width:${confidencePct}%; background: ${cat.color};"></div></div>
      <span class="confidence-value" style="color:${cat.color};">${confidencePct}%</span>
    </div>
    <div class="result-block">
      <h3>Почему эта категория</h3>
      <p>${explain}</p>
    </div>
    <div class="result-block recommendation">
      <h3>Как правильно утилизировать</h3>
      <p>${sorting}</p>
    </div>
  `;
}

function isSupportedFile(file) {
  if (ALLOWED_TYPES.includes(file.type)) return true;
  // fallback по расширению, если браузер не определил MIME-тип
  return /\.(jpe?g|png|webp)$/i.test(file.name);
}

// Возвращает { element, url }: element — canvas/img с уже применённым EXIF-поворотом,
// на нём же строится превью, чтобы пользователь видел ровно то, что анализирует модель.
async function loadOrientedImageSource(file) {
  if ('createImageBitmap' in window) {
    try {
      const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
      const canvas = document.createElement('canvas');
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      canvas.getContext('2d').drawImage(bitmap, 0, 0);
      bitmap.close();
      return { element: canvas, url: canvas.toDataURL('image/jpeg', 0.92) };
    } catch (err) {
      console.warn('createImageBitmap не смог применить ориентацию EXIF, используем запасной способ:', err);
    }
  }

  const url = URL.createObjectURL(file);
  const img = new Image();
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = reject;
    img.src = url;
  });
  return { element: img, url };
}

async function handleFile(file) {
  if (!file) return;
  if (!isSupportedFile(file)) {
    showToast('Поддерживаются только форматы JPG, JPEG, PNG и WEBP');
    return;
  }

  showScreen('analyze');
  els.analyzeText.textContent = 'Анализируем изображение…';

  const analysisStart = Date.now();

  try {
    // Телефоны сохраняют поворот кадра как EXIF-флаг, а не поворачивают сами пиксели.
    // createImageBitmap(..., { imageOrientation: 'from-image' }) разворачивает кадр так,
    // как его видит человек, — иначе модель получает "лежащий на боку" объект и не узнаёт его.
    const source = await loadOrientedImageSource(file);
    els.previewImg.src = source.url;
    els.resultImg.src = source.url;

    const classification = await classifyImage(source.element);

    const elapsed = Date.now() - analysisStart;
    if (elapsed < MIN_ANALYZE_MS) {
      await new Promise(r => setTimeout(r, MIN_ANALYZE_MS - elapsed));
    }

    renderResult(classification);
    showScreen('result');
  } catch (err) {
    console.error(err);
    showToast('Не удалось обработать изображение. Попробуйте ещё раз.');
    showScreen('upload');
  }
}

// --- Обработчики событий ---

els.uploadBtn.addEventListener('click', () => els.fileInput.click());
els.dropzone.addEventListener('click', (e) => {
  if (e.target === els.uploadBtn) return;
  els.fileInput.click();
});

els.fileInput.addEventListener('change', (e) => {
  const file = e.target.files && e.target.files[0];
  handleFile(file);
  els.fileInput.value = '';
});

['dragenter', 'dragover'].forEach(evt => {
  els.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    els.dropzone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach(evt => {
  els.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    els.dropzone.classList.remove('dragover');
  });
});

els.dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  handleFile(file);
});
 
els.resetBtn.addEventListener('click', () => {
  els.previewImg.src = '';
  els.resultImg.src = '';
  showScreen('upload');
});
