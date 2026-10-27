(() => {
  const input = document.querySelector('#imageInput');
  const dropzone = document.querySelector('#dropzone');
  const previewWrap = document.querySelector('#previewWrap');
  const preview = document.querySelector('#imagePreview');
  const form = document.querySelector('#predictionForm');
  const button = document.querySelector('#predictButton');

  function showImage(file) {
    if (!file || !['image/jpeg', 'image/png'].includes(file.type) || file.size > 8 * 1024 * 1024) {
      window.alert('Choose a JPG or PNG image no larger than 8 MB.');
      return;
    }
    preview.src = URL.createObjectURL(file);
    previewWrap.classList.remove('d-none');
  }

  if (input) {
    input.addEventListener('change', () => showImage(input.files[0]));
    document.querySelector('#removeImage').addEventListener('click', () => {
      input.value = '';
      preview.src = '';
      previewWrap.classList.add('d-none');
    });
    ['dragenter', 'dragover'].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add('is-dragging');
    }));
    ['dragleave', 'drop'].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove('is-dragging');
    }));
    dropzone.addEventListener('drop', (event) => {
      const [file] = event.dataTransfer.files;
      if (file) {
        const transfer = new DataTransfer();
        transfer.items.add(file);
        input.files = transfer.files;
        showImage(file);
      }
    });
  }

  if (form) form.addEventListener('submit', () => {
    if (input.files.length && button) {
      button.disabled = true;
      button.innerHTML = '<span class="spinner-border spinner-border-sm me-2" aria-hidden="true"></span>Analyzing image…';
    }
  });

  const charts = window.cropSenseCharts;
  if (!charts || !window.Chart) return;
  const palette = ['#295f48', '#76a94b', '#d5a53d', '#5b8d89', '#a25c3e', '#99bd72'];
  const distribution = document.querySelector('#distributionChart');
  if (distribution) new Chart(distribution, {
    type: 'doughnut',
    data: { labels: charts.labels, datasets: [{ data: charts.values, backgroundColor: palette, borderWidth: 0, hoverOffset: 5 }] },
    options: { maintainAspectRatio: false, cutout: '66%', plugins: { legend: { position: 'bottom', labels: { usePointStyle: true, boxWidth: 8, color: '#52625a', padding: 15, font: { family: 'DM Sans' } } } } }
  });
  const activity = document.querySelector('#activityChart');
  if (activity) new Chart(activity, {
    type: 'bar',
    data: { labels: charts.months, datasets: [{ data: charts.activity, backgroundColor: '#76a94b', borderRadius: 3, maxBarThickness: 34 }] },
    options: { maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { display: false }, ticks: { color: '#65736a' } }, y: { beginAtZero: true, ticks: { precision: 0, color: '#65736a' }, grid: { color: '#e8ece6' } } } }
  });
})();