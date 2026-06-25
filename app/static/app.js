const resultOutput = document.getElementById('resultOutput');
const memoryList = document.getElementById('memoryList');
const uploadResult = document.getElementById('uploadResult');

const requestJson = async (url, options) => {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
};

const renderJson = (data) => JSON.stringify(data, null, 2);

const loadMemory = async () => {
  try {
    const memories = await requestJson('/api/memory');
    memoryList.textContent = renderJson(memories);
  } catch (error) {
    memoryList.textContent = `Error loading memory: ${error.message}`;
  }
};

const runResearch = async () => {
  const question = document.getElementById('question').value.trim();
  const use_web_search = document.getElementById('webSearch').checked;
  const use_pdf_search = document.getElementById('pdfSearch').checked;
  const use_memory = document.getElementById('memorySearch').checked;

  if (!question) {
    resultOutput.textContent = 'Please enter a research query.';
    return;
  }

  resultOutput.textContent = 'Running research...';

  try {
    const payload = {
      question,
      use_web_search,
      use_pdf_search,
      use_memory,
    };

    const result = await requestJson('/api/research', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    resultOutput.textContent = renderJson(result);
  } catch (error) {
    resultOutput.textContent = `Research error: ${error.message}`;
  }
};

const uploadPdf = async () => {
  const fileInput = document.getElementById('pdfUpload');
  const file = fileInput.files[0];
  if (!file) {
    uploadResult.textContent = 'Please select a PDF file to upload.';
    return;
  }

  const form = new FormData();
  form.append('file', file);
  uploadResult.textContent = 'Uploading PDF...';

  try {
    const result = await requestJson('/api/upload-pdf', {
      method: 'POST',
      body: form,
    });
    uploadResult.textContent = renderJson(result);
  } catch (error) {
    uploadResult.textContent = `Upload error: ${error.message}`;
  }
};

const addMemory = async () => {
  const id = document.getElementById('memoryId').value.trim();
  const content = document.getElementById('memoryContent').value.trim();
  if (!id || !content) {
    memoryList.textContent = 'Memory ID and content are required.';
    return;
  }

  try {
    const payload = { id, content, tags: [] };
    const result = await requestJson('/api/memory', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    memoryList.textContent = `Added memory:\n${renderJson(result)}`;
    await loadMemory();
  } catch (error) {
    memoryList.textContent = `Memory error: ${error.message}`;
  }
};

window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('runResearch').addEventListener('click', runResearch);
  document.getElementById('uploadPdf').addEventListener('click', uploadPdf);
  document.getElementById('addMemory').addEventListener('click', addMemory);
  document.getElementById('refreshMemory').addEventListener('click', loadMemory);
  loadMemory();
});
