(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const MAX_IMAGE_SIZE = 15 * 1024 * 1024;
  const ALLOWED_IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/webp", "image/bmp"]);
  const ALLOWED_IMAGE_EXTENSIONS = new Set(["jpg", "jpeg", "png", "webp", "bmp"]);
  const state = { stream: null, contacts: [], searchTimer: null, previewUrl: null };

  const elements = {
    camera: $("#camera"),
    cameraStage: $("#camera-stage"),
    cameraToggle: $("#camera-toggle"),
    captureButton: $("#capture-button"),
    canvas: $("#capture-canvas"),
    preview: $("#captured-preview"),
    upload: $("#image-upload"),
    uploadDropzone: $("#upload-dropzone"),
    processing: $("#processing"),
    detectionNote: $("#detection-note"),
    form: $("#contact-form"),
    contactId: $("#contact-id"),
    imageToken: $("#image-token"),
    rawText: $("#raw-text"),
    search: $("#contact-search"),
    contactsBody: $("#contacts-body"),
    emptyContacts: $("#empty-contacts"),
    contactCount: $("#contact-count"),
    ocrStatus: $("#ocr-status"),
    verificationPanel: $("#verification-panel"),
    scoreCards: $("#score-cards"),
    verificationChecks: $("#verification-checks"),
    duplicateNote: $("#duplicate-note"),
  };

  function toast(message, type = "success", duration = 3800) {
    const item = document.createElement("div");
    item.className = `toast ${type}`;
    item.textContent = message;
    $("#toast-region").append(item);
    window.setTimeout(() => item.remove(), duration);
  }

  async function api(url, options = {}) {
    const response = await fetch(url, options);
    const result = await response.json().catch(() => ({
      ok: false,
      error: { message: "서버 응답을 읽지 못했습니다." },
    }));
    if (!response.ok || result.ok === false) {
      const error = new Error(result.error?.message || "요청에 실패했습니다.");
      error.code = result.error?.code;
      error.details = result.error;
      throw error;
    }
    return result;
  }

  function switchView(name) {
    $$(".view").forEach((view) => view.classList.toggle("active", view.id === `${name}-view`));
    $$(".nav-button").forEach((button) => button.classList.toggle("active", button.dataset.view === name));
    if (name === "contacts") loadContacts();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function checkHealth() {
    try {
      const result = await api("/api/health");
      elements.ocrStatus.classList.remove("checking", "ready", "error");
      if (result.ocr.installed) {
        if (result.ocr.warming) {
          elements.ocrStatus.classList.add("checking");
          elements.ocrStatus.lastChild.textContent = "OCR 모델 준비 중";
          window.setTimeout(checkHealth, 2000);
        } else if (result.ocr.error && !result.ocr.ready) {
          elements.ocrStatus.classList.add("error");
          elements.ocrStatus.lastChild.textContent = "OCR 준비 실패";
          elements.ocrStatus.title = result.ocr.error;
        } else {
          elements.ocrStatus.classList.add("ready");
          elements.ocrStatus.lastChild.textContent = "OCR 준비됨";
          elements.ocrStatus.title = "";
        }
      } else {
        elements.ocrStatus.classList.add("error");
        elements.ocrStatus.lastChild.textContent = "OCR 설치 필요";
        elements.ocrStatus.title = "requirements.txt 설치 후 OCR을 사용할 수 있습니다.";
      }
    } catch {
      elements.ocrStatus.classList.add("error");
      elements.ocrStatus.lastChild.textContent = "서버 연결 오류";
    }
  }

  async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      toast("이 브라우저는 카메라 촬영을 지원하지 않습니다. 이미지 선택을 이용해 주세요.", "error");
      return;
    }
    try {
      state.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 3840 },
          height: { ideal: 2160 },
          advanced: [{ focusMode: "continuous" }],
        },
        audio: false,
      });
      elements.camera.srcObject = state.stream;
      elements.cameraStage.className = "camera-stage camera-active";
      elements.cameraToggle.textContent = "카메라 끄기";
      elements.captureButton.disabled = false;
    } catch (error) {
      toast(`카메라를 열 수 없습니다: ${error.message}`, "error");
    }
  }

  function stopCamera() {
    state.stream?.getTracks().forEach((track) => track.stop());
    state.stream = null;
    elements.camera.srcObject = null;
    elements.cameraStage.className = "camera-stage";
    elements.cameraToggle.textContent = "카메라 시작";
    elements.captureButton.disabled = true;
  }

  async function captureFrame() {
    const track = state.stream?.getVideoTracks?.()[0];
    if (track && "ImageCapture" in window) {
      try {
        const photo = await new ImageCapture(track).takePhoto();
        await processImage(new File([photo], "camera-capture.jpg", { type: photo.type || "image/jpeg" }));
        return;
      } catch {
        // Some desktop cameras expose ImageCapture but do not support takePhoto.
      }
    }
    const { videoWidth, videoHeight } = elements.camera;
    if (!videoWidth || !videoHeight) return;
    elements.canvas.width = videoWidth;
    elements.canvas.height = videoHeight;
    elements.canvas.getContext("2d").drawImage(elements.camera, 0, 0);
    const blob = await new Promise((resolve) => elements.canvas.toBlob(resolve, "image/jpeg", 0.94));
    if (blob) await processImage(new File([blob], "camera-capture.jpg", { type: "image/jpeg" }));
  }

  function fillForm(fields) {
    for (const [key, value] of Object.entries(fields || {})) {
      const field = elements.form.elements.namedItem(key);
      if (field) field.value = value || "";
    }
  }

  function renderFieldConfidence(confidence = {}) {
    $$('[data-confidence]').forEach((badge) => {
      const value = confidence[badge.dataset.confidence]?.confidence;
      badge.textContent = Number.isFinite(value) ? `${Math.round(value * 100)}%` : "";
      badge.className = Number.isFinite(value) ? (value < 0.65 ? "confidence-low" : "confidence-ok") : "";
    });
  }

  function renderVerification(verification) {
    if (!verification?.scores) {
      elements.verificationPanel.hidden = true;
      return;
    }
    const { accuracy, consistency, safety } = verification.scores;
    const safetyLabel = { ok: "확인됨", unknown: "확인 불가", warn: "주의" }[safety] || "확인 불가";
    elements.scoreCards.innerHTML = `
      <div><span>인식 정확도</span><strong>${accuracy || 0}<small>점</small></strong></div>
      <div><span>정보 정합성</span><strong>${consistency || 0}<small>점</small></strong></div>
      <div class="score-${escapeHtml(safety)}"><span>안전성</span><strong>${escapeHtml(safetyLabel)}</strong></div>`;
    elements.verificationChecks.innerHTML = (verification.checks || []).map((check) => `
      <li class="check-${escapeHtml(check.state)}"><i></i><div><strong>${escapeHtml(check.label)}</strong><span>${escapeHtml(check.message)}</span>${check.suggestion ? `<button class="apply-suggestion" data-field="${escapeHtml(check.field)}" data-value="${escapeHtml(check.suggestion)}" type="button">${escapeHtml(check.suggestion)} 적용</button>` : ""}</div></li>`).join("");
    elements.duplicateNote.hidden = !verification.duplicate;
    elements.duplicateNote.textContent = verification.duplicate ? `유사한 기존 고객 #${verification.duplicate.contact_id} · ${Math.round(verification.duplicate.similarity * 100)}% (${verification.duplicate.reason})` : "";
    elements.verificationPanel.hidden = false;
  }

  function validateImageFile(file) {
    if (!(file instanceof File)) return "업로드할 이미지 파일을 선택해 주세요.";
    if (file.size <= 0) return "내용이 없는 파일은 업로드할 수 없습니다.";
    if (file.size > MAX_IMAGE_SIZE) return "15MB 이하의 이미지를 선택해 주세요.";
    const extension = file.name.includes(".") ? file.name.split(".").pop().toLowerCase() : "";
    if ((file.type && !ALLOWED_IMAGE_TYPES.has(file.type)) || (extension && !ALLOWED_IMAGE_EXTENSIONS.has(extension))) {
      return "JPG, PNG, WEBP, BMP 이미지 파일만 업로드할 수 있습니다.";
    }
    if (!file.type && !extension) return "이미지 파일 형식을 확인할 수 없습니다.";
    return "";
  }

  function clearLocalPreviewUrl() {
    if (!state.previewUrl) return;
    URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = null;
  }

  function showLocalPreview(file) {
    clearLocalPreviewUrl();
    state.previewUrl = URL.createObjectURL(file);
    elements.preview.src = state.previewUrl;
    elements.cameraStage.className = "camera-stage preview-active upload-preview";
  }

  async function uploadImage(file) {
    const problem = validateImageFile(file);
    if (problem) {
      toast(problem, "error");
      elements.upload.value = "";
      return;
    }
    stopCamera();
    showLocalPreview(file);
    elements.detectionNote.hidden = false;
    elements.detectionNote.textContent = `${file.name} 업로드 완료 · OCR 분석을 시작합니다.`;
    await processImage(file);
  }

  async function processImage(file) {
    const problem = validateImageFile(file);
    if (problem) {
      toast(problem, "error");
      return;
    }
    const formData = new FormData();
    formData.append("image", file, file.name || "business-card.jpg");
    elements.processing.classList.add("active");
    try {
      const result = await api("/api/ocr", { method: "POST", body: formData });
      const data = result.data;
      stopCamera();
      clearLocalPreviewUrl();
      elements.preview.src = data.preview;
      elements.cameraStage.className = "camera-stage preview-active";
      elements.imageToken.value = data.image_token;
      elements.contactId.value = "";
      fillForm(data.fields);
      renderFieldConfidence(data.field_confidence);
      renderVerification(data.verification);
      const notes = data.detection.warnings || [];
      elements.detectionNote.hidden = notes.length === 0;
      elements.detectionNote.textContent = notes.join(" ");
      toast(`${data.ocr_lines.length}개 텍스트 영역을 인식했습니다.`);
    } catch (error) {
      if (state.previewUrl) {
        elements.detectionNote.hidden = false;
        elements.detectionNote.textContent = `이미지를 불러왔지만 OCR 처리에 실패했습니다: ${error.message}`;
      }
      toast(error.message, "error", 6500);
    } finally {
      elements.processing.classList.remove("active");
      elements.upload.value = "";
    }
  }

  function formPayload() {
    const data = Object.fromEntries(new FormData(elements.form).entries());
    data.image_token = elements.imageToken.value;
    return data;
  }

  async function saveContact(event, allowDuplicate = false) {
    event?.preventDefault();
    const data = formPayload();
    if (allowDuplicate) data.allow_duplicate = true;
    const id = elements.contactId.value;
    const url = id ? `/api/contacts/${id}` : "/api/contacts";
    try {
      await api(url, {
        method: id ? "PUT" : "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      toast(id ? "고객정보를 수정했습니다." : "고객정보를 저장했습니다.");
      resetForm();
      switchView("contacts");
    } catch (error) {
      if (error.code === "DUPLICATE_CONTACT") {
        const duplicate = error.details.duplicates?.[0];
        const label = duplicate ? `${duplicate.name || "이름 없음"} (${duplicate.company || "회사 없음"})` : "기존 고객";
        if (window.confirm(`${label}과 전화번호 또는 이메일이 같습니다. 그래도 저장할까요?`)) {
          return saveContact(null, true);
        }
      } else {
        toast(error.message, "error");
      }
    }
  }

  function resetForm() {
    elements.form.reset();
    elements.form.querySelectorAll("input, textarea").forEach((field) => {
      if (field.type === "checkbox" || field.type === "radio") field.checked = false;
      else field.value = "";
    });
    elements.contactId.value = "";
    elements.imageToken.value = "";
    elements.rawText.value = "";
    elements.upload.value = "";
    clearLocalPreviewUrl();
    elements.preview.src = "";
    elements.preview.removeAttribute("src");
    elements.cameraStage.className = state.stream
      ? "camera-stage camera-active"
      : "camera-stage";
    elements.processing.classList.remove("active");
    elements.detectionNote.hidden = true;
    elements.detectionNote.textContent = "";
    renderFieldConfidence();
    renderVerification(null);
    const rawDetails = elements.rawText.closest("details");
    if (rawDetails) rawDetails.open = false;
    elements.canvas.width = 0;
    elements.canvas.height = 0;
    $("#save-button").lastChild.textContent = " 고객정보 저장";
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>'"]/g, (char) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
    })[char]);
  }

  function initials(name) {
    const value = String(name || "?").trim();
    return escapeHtml(value.slice(0, 2).toUpperCase());
  }

  function renderContacts(contacts) {
    state.contacts = contacts;
    elements.contactCount.textContent = contacts.length;
    elements.emptyContacts.classList.toggle("visible", contacts.length === 0);
    elements.contactsBody.innerHTML = contacts.map((contact) => `
      <tr>
        <td><div class="person-cell"><span class="avatar">${initials(contact.name)}</span><span>${escapeHtml(contact.name || "이름 없음")}</span></div></td>
        <td>${escapeHtml(contact.company || "-")}<span class="subtle">${escapeHtml(contact.job_title || "")}</span></td>
        <td>${escapeHtml(contact.phone || "-")}<span class="subtle">${escapeHtml(contact.phone2 || "")}</span><span class="subtle">${contact.fax ? `팩스 ${escapeHtml(contact.fax)}` : ""}</span></td>
        <td>${escapeHtml(contact.email || "-")}</td>
        <td>${escapeHtml((contact.created_at || "").slice(0, 10))}</td>
        <td><div class="row-actions">
          <button class="icon-button edit-contact" data-id="${contact.id}" type="button">수정</button>
          <button class="icon-button danger delete-contact" data-id="${contact.id}" type="button">삭제</button>
        </div></td>
      </tr>`).join("");
  }

  async function loadContacts() {
    try {
      const query = encodeURIComponent(elements.search.value.trim());
      const result = await api(`/api/contacts?q=${query}`);
      renderContacts(result.data);
      $("#csv-export").href = `/api/export?format=csv&q=${query}`;
      $("#xlsx-export").href = `/api/export?format=xlsx&q=${query}`;
    } catch (error) {
      toast(error.message, "error");
    }
  }

  async function editContact(id) {
    try {
      const result = await api(`/api/contacts/${id}`);
      resetForm();
      fillForm(result.data);
      elements.contactId.value = result.data.id;
      elements.imageToken.value = result.data.image_token || "";
      if (result.data.image_token) {
        elements.preview.src = `/api/scans/${result.data.image_token}`;
        elements.cameraStage.className = "camera-stage preview-active";
      }
      let storedChecks = [];
      try { storedChecks = JSON.parse(result.data.verify_json || "[]"); } catch { storedChecks = []; }
      renderVerification({ scores: { accuracy: result.data.score_accuracy, consistency: result.data.score_consistency, safety: result.data.score_safety || "unknown" }, checks: storedChecks, duplicate: null });
      $("#save-button").lastChild.textContent = " 고객정보 수정";
      switchView("scanner");
    } catch (error) {
      toast(error.message, "error");
    }
  }

  async function deleteContact(id) {
    const contact = state.contacts.find((item) => item.id === Number(id));
    if (!window.confirm(`${contact?.name || "이 고객"}의 정보를 삭제할까요?`)) return;
    try {
      await api(`/api/contacts/${id}`, { method: "DELETE" });
      toast("고객정보를 삭제했습니다.");
      loadContacts();
    } catch (error) {
      toast(error.message, "error");
    }
  }

  async function reparseText() {
    try {
      const result = await api("/api/parse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ raw_text: elements.rawText.value }),
      });
      fillForm(result.data.fields);
      renderVerification(result.data.verification);
      toast("OCR 원문을 다시 분류했습니다.");
    } catch (error) {
      toast(error.message, "error");
    }
  }

  $$(".nav-button").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.view)));
  elements.cameraToggle.addEventListener("click", () => state.stream ? stopCamera() : startCamera());
  elements.captureButton.addEventListener("click", captureFrame);
  elements.upload.addEventListener("change", () => elements.upload.files[0] && uploadImage(elements.upload.files[0]));
  elements.uploadDropzone.addEventListener("click", () => elements.upload.click());
  elements.uploadDropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      elements.upload.click();
    }
  });
  elements.cameraStage.addEventListener("dragover", (event) => {
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
    elements.cameraStage.classList.add("upload-dragging");
  });
  elements.cameraStage.addEventListener("dragleave", (event) => {
    if (!elements.cameraStage.contains(event.relatedTarget)) elements.cameraStage.classList.remove("upload-dragging");
  });
  elements.cameraStage.addEventListener("drop", (event) => {
    event.preventDefault();
    elements.cameraStage.classList.remove("upload-dragging");
    const file = event.dataTransfer?.files?.[0];
    if (file) uploadImage(file);
  });
  elements.form.addEventListener("submit", saveContact);
  elements.verificationChecks.addEventListener("click", (event) => {
    const button = event.target.closest(".apply-suggestion");
    if (!button) return;
    const field = elements.form.elements.namedItem(button.dataset.field);
    if (field) field.value = button.dataset.value;
  });
  $("#form-reset").addEventListener("click", resetForm);
  $("#reparse-button").addEventListener("click", reparseText);
  $("#new-contact").addEventListener("click", () => { resetForm(); switchView("scanner"); });
  elements.search.addEventListener("input", () => {
    window.clearTimeout(state.searchTimer);
    state.searchTimer = window.setTimeout(loadContacts, 250);
  });
  elements.contactsBody.addEventListener("click", (event) => {
    const edit = event.target.closest(".edit-contact");
    const remove = event.target.closest(".delete-contact");
    if (edit) editContact(edit.dataset.id);
    if (remove) deleteContact(remove.dataset.id);
  });
  window.addEventListener("beforeunload", stopCamera);

  checkHealth();
  loadContacts();
})();
