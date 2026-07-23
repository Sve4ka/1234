/**
 * Утилиты для обработки edge-кейсов
 */

(function() {
  'use strict';

  // ============ ВАЛИДАЦИЯ ДАННЫХ ============

  /**
   * Валидация структуры данных предсказаний
   * Ожидаемый формат: [{hash: string, semester: number, debt: number}, ...]
   */
  window.validatePredictionData = function(data) {
    const errors = [];
    const warnings = [];

    // Проверка типа
    if (!Array.isArray(data)) {
      return { valid: false, errors: ['Ожидался массив данных, получен: ' + typeof data], warnings: [], data: [] };
    }

    // Пустой массив
    if (data.length === 0) {
      return { valid: false, errors: ['Ответ API пуст — нет данных для отображения'], warnings: [], data: [] };
    }

    const validRecords = [];
    let skippedCount = 0;

    data.forEach((record, index) => {
      // Проверка объекта
      if (!record || typeof record !== 'object') {
        errors.push(`Запись #${index + 1}: не является объектом`);
        skippedCount++;
        return;
      }

      // Проверка hash
      if (!record.hash || typeof record.hash !== 'string') {
        warnings.push(`Запись #${index + 1}: отсутствует или невалидный hash`);
        record.hash = 'unknown_' + index;
      }

      // Проверка semester
      if (record.semester === undefined || record.semester === null) {
        errors.push(`Запись #${index + 1} (${record.hash}): отсутствует семестр`);
        skippedCount++;
        return;
      }
      const semester = parseInt(record.semester);
      if (isNaN(semester) || semester < 1 || semester > 12) {
        warnings.push(`Запись #${index + 1}: невалидный семестр (${record.semester}), пропущена`);
        skippedCount++;
        return;
      }

      // Проверка debt
      if (record.debt === undefined || record.debt === null) {
        warnings.push(`Запись #${index + 1}: отсутствует количество долгов, установлено 0`);
        record.debt = 0;
      }
      const debt = parseInt(record.debt);
      if (isNaN(debt) || debt < 0) {
        warnings.push(`Запись #${index + 1}: невалидное количество долгов (${record.debt})`);
        record.debt = 0;
      }

      validRecords.push({
        hash: String(record.hash),
        semester: semester,
        debt: Math.min(debt, 3) // 3+ объединяем
      });
    });

    return {
      valid: validRecords.length > 0,
      errors,
      warnings,
      data: validRecords,
      skippedCount
    };
  };

  // ============ FETCH С ТАЙМАУТОМ ============

  /**
   * Fetch с таймаутом и обработкой offline
   */
  window.fetchWithTimeout = async function(url, options = {}, timeout = 300000) {
    // Проверка соединения
    if (!navigator.onLine) {
      throw new Error('Нет подключения к интернету. Проверьте сеть и попробуйте снова.');
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`Сервер вернул ошибку: ${response.status} ${response.statusText}`);
      }

      return response;
    } catch (error) {
      clearTimeout(timeoutId);
      if (error.name === 'AbortError') {
        throw new Error(`Превышено время ожидания (${timeout / 1000} сек). Сервер не отвечает.`);
      }
      throw error;
    }
  };

  // ============ OFFLINE-ИНДИКАТОР ============

  /**
   * Показать/скрыть индикатор offline
   */
  window.setupOfflineIndicator = function() {
    let indicator = document.getElementById('offline-indicator');
    if (!indicator) {
      indicator = document.createElement('div');
      indicator.id = 'offline-indicator';
      indicator.innerHTML = '📡 Нет подключения к сети';
      document.body.appendChild(indicator);
    }

    function updateStatus() {
      if (navigator.onLine) {
        indicator.classList.remove('visible');
        showToast('Подключение восстановлено', 'success', 2000);
      } else {
        indicator.classList.add('visible');
        showToast('Подключение к сети потеряно', 'warning', 0, { sticky: true });
      }
    }

    window.addEventListener('online', updateStatus);
    window.addEventListener('offline', updateStatus);

    // Начальная проверка
    if (!navigator.onLine) {
      indicator.classList.add('visible');
    }
  };

  // ============ SKELETON LOADER ============

  /**
   * Показать skeleton-загрузку
   */
  window.showSkeleton = function(containerId, rows = 5, cols = 4) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';
    container.classList.add('skeleton-active');

    for (let i = 0; i < rows; i++) {
      const row = document.createElement('div');
      row.className = 'skeleton-row';
      for (let j = 0; j < cols; j++) {
        const cell = document.createElement('div');
        cell.className = 'skeleton-cell';
        cell.style.animationDelay = `${(i * cols + j) * 0.05}s`;
        row.appendChild(cell);
      }
      container.appendChild(row);
    }
  };

  window.hideSkeleton = function(containerId) {
    const container = document.getElementById(containerId);
    if (container) container.classList.remove('skeleton-active');
  };

  // ============ EMPTY STATE ============

  /**
   * Показать пустое состояние
   */
  window.showEmptyState = function(containerId, icon, title, description) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">${icon}</div>
        <div class="empty-state-title">${title}</div>
        <div class="empty-state-desc">${description}</div>
      </div>
    `;
  };
})();