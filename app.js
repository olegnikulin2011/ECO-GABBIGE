// ECObage — классификация мусора по фото с помощью MobileNet (TensorFlow.js)

const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
// ImageNet has only one apple-like class ("Granny Smith"), one bottle-type per material, etc.,
// so a correct match can still come back with modest probability (e.g. a red apple vs the
// green-apple reference class). Gating on a high top-1 confidence caused correct matches to be
// discarded as "uncertain" — so we only require a rule match, with a low floor to filter noise.
const MIN_CONFIDENCE = 0.12;
const MIN_ANALYZE_MS = 1400; // чтобы анимация анализа была заметна даже при быстрой модели

const CATEGORIES = {
  plastic: {
    label: 'Пластик',
    emoji: '🟡',
    color: '#f2b705',
    bg: '#fef7de',
    explain: 'Предмет изготовлен из пластика — материала, который разлагается сотни лет, но хорошо подходит для вторичной переработки.',
    sorting: 'Промойте от остатков пищи, при возможности снимите этикетку и выбросьте в контейнер для пластика.'
  },
  glass: {
    label: 'Стекло',
    emoji: '🟢',
    color: '#2e9e5b',
    bg: '#e6f6ec',
    explain: 'Предмет сделан из стекла — материала, который можно перерабатывать неограниченное количество раз без потери качества.',
    sorting: 'Снимите крышку и промойте от остатков. Не разбивайте — целое стекло перерабатывать проще и безопаснее.'
  },
  organic: {
    label: 'Органика',
    emoji: '🟤',
    color: '#8a5a34',
    bg: '#f3e9e0',
    explain: 'Это пищевые или растительные отходы. Они разлагаются естественным образом и отлично подходят для компостирования.',
    sorting: 'Выбросьте в контейнер для органических отходов или используйте для домашнего компоста.'
  },
  paper: {
    label: 'Бумага',
    emoji: '🔵',
    color: '#2f7fe0',
    bg: '#e8f1fc',
    explain: 'Предмет изготовлен из бумаги или картона — перерабатываемого материала на основе целлюлозных волокон.',
    sorting: 'Уберите скрепки, плёнку и скотч, сложите и выбросьте в контейнер для бумаги и картона. Не мочите бумагу.'
  },
  metal: {
    label: 'Металл',
    emoji: '⚪',
    color: '#7d868c',
    bg: '#eef1f2',
    explain: 'Предмет содержит металл, который отлично поддаётся переплавке и практически бесконечному повторному использованию.',
    sorting: 'Промойте от остатков пищи, при необходимости сомните для экономии места и выбросьте в контейнер для металла.'
  }
};

// Правила сопоставления классов MobileNet (ImageNet) с категориями отходов.
// Проверяются по порядку — первое совпадение побеждает.
const RULES = [
  { kw: 'water bottle', cat: 'plastic', ru: 'Пластиковая бутылка (вода)' },
  { kw: 'pop bottle', cat: 'plastic', ru: 'Пластиковая бутылка (напиток)' },
  { kw: 'soda bottle', cat: 'plastic', ru: 'Пластиковая бутылка (напиток)' },
  { kw: 'pill bottle', cat: 'plastic', ru: 'Пластиковый флакон от таблеток' },
  { kw: 'plastic bag', cat: 'plastic', ru: 'Пластиковый пакет' },
  { kw: 'shower cap', cat: 'plastic', ru: 'Пластиковая шапочка для душа' },
  { kw: 'water jug', cat: 'plastic', ru: 'Пластиковая канистра' },

  { kw: 'wine bottle', cat: 'glass', ru: 'Стеклянная бутылка (вино)' },
  { kw: 'beer bottle', cat: 'glass', ru: 'Стеклянная бутылка (пиво)' },
  { kw: 'beer glass', cat: 'glass', ru: 'Стеклянный бокал' },
  { kw: 'goblet', cat: 'glass', ru: 'Стеклянный бокал' },
  { kw: 'vase', cat: 'glass', ru: 'Стеклянная ваза / банка' },
  { kw: 'perfume', cat: 'glass', ru: 'Стеклянный флакон парфюма' },
  { kw: 'beaker', cat: 'glass', ru: 'Стеклянная колба' },

  { kw: 'tin can', cat: 'metal', ru: 'Жестяная банка' },
  { kw: 'milk can', cat: 'metal', ru: 'Металлический бидон' },
  { kw: 'nail', cat: 'metal', ru: 'Металлический гвоздь' },
  { kw: 'screw', cat: 'metal', ru: 'Металлический шуруп' },
  { kw: 'safety pin', cat: 'metal', ru: 'Металлическая булавка' },
  { kw: 'padlock', cat: 'metal', ru: 'Металлический замок' },
  { kw: 'chain', cat: 'metal', ru: 'Металлическая цепь' },
  { kw: 'lighter', cat: 'metal', ru: 'Металлическая зажигалка' },

  { kw: 'envelope', cat: 'paper', ru: 'Бумажный конверт' },
  { kw: 'carton', cat: 'paper', ru: 'Картонная упаковка' },
  { kw: 'paper towel', cat: 'paper', ru: 'Бумажное полотенце' },
  { kw: 'comic book', cat: 'paper', ru: 'Газета / журнал' },
  { kw: 'menu', cat: 'paper', ru: 'Бумажное меню' },
  { kw: 'binder', cat: 'paper', ru: 'Бумажная папка' },
  { kw: 'book jacket', cat: 'paper', ru: 'Бумажная обложка книги' },

  { kw: 'banana', cat: 'organic', ru: 'Банановая кожура' },
  { kw: 'orange', cat: 'organic', ru: 'Апельсиновая корка' },
  { kw: 'lemon', cat: 'organic', ru: 'Лимон' },
  { kw: 'strawberry', cat: 'organic', ru: 'Клубника' },
  { kw: 'pineapple', cat: 'organic', ru: 'Ананас' },
  { kw: 'fig', cat: 'organic', ru: 'Инжир' },
  { kw: 'pomegranate', cat: 'organic', ru: 'Гранат' },
  { kw: 'artichoke', cat: 'organic', ru: 'Артишок' },
  { kw: 'cucumber', cat: 'organic', ru: 'Огурец' },
  { kw: 'zucchini', cat: 'organic', ru: 'Кабачок' },
  { kw: 'bell pepper', cat: 'organic', ru: 'Болгарский перец' },
  { kw: 'mushroom', cat: 'organic', ru: 'Гриб' },
  { kw: 'corn', cat: 'organic', ru: 'Кукурузный початок' },
  { kw: 'cauliflower', cat: 'organic', ru: 'Цветная капуста' },
  { kw: 'broccoli', cat: 'organic', ru: 'Брокколи' },
  { kw: 'head cabbage', cat: 'organic', ru: 'Капуста' },
  { kw: 'squash', cat: 'organic', ru: 'Тыква / кабачок' },
  { kw: 'granny smith', cat: 'organic', ru: 'Яблоко' },
  { kw: 'custard apple', cat: 'organic', ru: 'Фрукт (черимойя)' },
  { kw: 'jackfruit', cat: 'organic', ru: 'Джекфрут' }
];

const PERSON_MIN_SCORE = 0.5;

let mobilenetModel = null;
let modelLoadingPromise = null;
let cocoSsdModel = null;
let cocoSsdLoadingPromise = null;

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

function getModel() {
  if (mobilenetModel) return Promise.resolve(mobilenetModel);
  if (!modelLoadingPromise) {
    modelLoadingPromise = mobilenet.load({ version: 2, alpha: 1.0 }).then(m => {
      mobilenetModel = m;
      return m;
    });
  }
  return modelLoadingPromise;
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

// Начинаем загрузку моделей сразу при открытии страницы, чтобы первый анализ был быстрее.
// MobileNet (ImageNet-1000) отвечает за материал предмета, а COCO-SSD — за то, есть ли на
// фото человек: у ImageNet почти нет класса «человек» (только «бейсболист», «жених»,
// «аквалангист»), поэтому фото людей в других позах/ракурсах он не распознаёт как людей,
// а COCO-SSD обучен именно на классе "person" и не зависит от одежды или позы.
getModel().catch(err => console.error('Не удалось загрузить модель MobileNet:', err));
getCocoSsdModel().catch(err => console.error('Не удалось загрузить модель COCO-SSD:', err));

function matchRule(label) {
  const lower = label.toLowerCase();
  return RULES.find(r => lower.includes(r.kw));
}

async function classifyImage(imgEl) {
  const [model, detector] = await Promise.all([getModel(), getCocoSsdModel()]);

  const detections = await detector.detect(imgEl);
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

  const predictions = await model.classify(imgEl, 5);

  for (const pred of predictions) {
    const rule = matchRule(pred.className);
    if (rule) {
      return {
        matched: true,
        rule,
        confidence: pred.probability,
        rawLabel: pred.className
      };
    }
  }

  return {
    matched: false,
    confidence: predictions[0] ? predictions[0].probability : 0,
    rawLabel: predictions[0] ? predictions[0].className : ''
  };
}

function renderResult(classification) {
  const confidencePct = Math.round(classification.confidence * 100);

  if (!classification.matched || classification.confidence < MIN_CONFIDENCE) {
    els.resultCard.className = 'result-card uncertain';
    els.resultCard.innerHTML = `
      <div class="uncertain-icon">🤔</div>
      <p class="uncertain-text">Не удалось точно определить объект. Попробуйте сделать более качественную фотографию.</p>
    `;
    return;
  }

  const cat = CATEGORIES[classification.rule.cat];
  const explain = classification.isPerson
    ? 'Человек биологически тоже состоит из органики 😄 (это шутливое распознавание, а не серьёзная рекомендация).'
    : cat.explain;
  const sorting = classification.isPerson
    ? 'Пожалуйста, не выбрасывайте людей в мусор — просто улыбнитесь и сделайте фото чего-нибудь настоящего для сортировки 🙂'
    : cat.sorting;

  els.resultCard.className = 'result-card';
  els.resultCard.style.borderTopColor = cat.color;
  els.resultCard.innerHTML = `
    <div class="cat-badge" style="background:${cat.bg}; color:${cat.color};">
      <span class="cat-icon">${cat.emoji}</span> ${cat.label}
    </div>
    <h2 class="item-name">${classification.rule.ru}</h2>
    <div class="confidence-row">
      <span class="confidence-label">Вероятность</span>
      <div class="confidence-bar"><div class="confidence-fill" style="width:${confidencePct}%; background: linear-gradient(90deg, ${cat.color}, ${cat.color});"></div></div>
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
