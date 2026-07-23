/**
 * Toast-уведомления
 * Использование: showToast('Текст', 'success', 3000)
 * Типы: 'success', 'error', 'warning', 'info'
 */

(function() {
  'use strict';

  // Создаём контейнер для тостов (один на всю страницу)
  function getContainer() {
    let container = document.getElementById('toast-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'toast-container';
      document.body.appendChild(container);
    }
    return container;
  }

  // Иконки для разных типов
  const icons = {
    success: '✅',
    error: '',
    warning: '️',
    info: 'ℹ️'
  };

  /**
   * Показать toast-уведомление
   * @param {string} message - Текст сообщения
   * @param {string} type - Тип: success|error|warning|info
   * @param {number} duration - Время показа в мс (0 = не исчезает)
   * @param {Object} options - Доп. опции
   */
  window.showToast = function(message, type = 'info', duration = 4000, options = {}) {
    const container = getContainer();

    // Создаём элемент тоста
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    if (options.sticky) toast.classList.add('toast-sticky');

    toast.innerHTML = `
      <div class="toast-icon">${icons[type] || icons.info}</div>
      <div class="toast-content">
        ${options.title ? `<div class="toast-title">${options.title}</div>` : ''}
        <div class="toast-message">${message}</div>
      </div>
      <button class="toast-close" aria-label="Закрыть">✕</button>
      <div class="toast-progress"></div>
    `;

    container.appendChild(toast);

    // Анимация появления (через requestAnimationFrame для плавности)
    requestAnimationFrame(() => {
      toast.classList.add('toast-show');
    });

    // Кнопка закрытия
    const closeBtn = toast.querySelector('.toast-close');
    closeBtn.addEventListener('click', () => removeToast(toast));

    // Автоудаление
    let timeoutId = null;
    if (duration > 0 && !options.sticky) {
      timeoutId = setTimeout(() => removeToast(toast), duration);
    }

    // Пауза при наведении
    toast.addEventListener('mouseenter', () => {
      if (timeoutId) clearTimeout(timeoutId);
      toast.classList.add('toast-paused');
    });

    toast.addEventListener('mouseleave', () => {
      if (duration > 0 && !options.sticky) {
        timeoutId = setTimeout(() => removeToast(toast), duration / 2);
      }
      toast.classList.remove('toast-paused');
    });

    return toast;
  };

  // Удаление тоста с анимацией
  function removeToast(toast) {
    if (toast.classList.contains('toast-removing')) return;
    toast.classList.add('toast-removing');
    setTimeout(() => {
      if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 300);
  }

  // Удалить все тосты
  window.clearAllToasts = function() {
    const container = getContainer();
    container.querySelectorAll('.toast').forEach(t => removeToast(t));
  };
})();