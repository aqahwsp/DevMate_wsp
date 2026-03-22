"""FastAPI web interface for DevMate."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
)

from devmate.config import load_config
from devmate.logging_config import configure_logging
from devmate.runtime import DevMateRuntime, RunController
from devmate.schemas import (
    GenerateRequest,
    GenerateResponse,
    GenerationJobStartResponse,
    GenerationJobStatusResponse,
    UploadedInitialFile,
    UploadedInitialFileList,
    WorkspaceListing,
)

LOGGER = logging.getLogger(__name__)
_WEB_REPLY_LOG_INTERVAL_SECONDS = 0.5
_WEB_REPLY_LOG_MIN_LENGTH_DELTA = 24
_WEB_REPLY_PERSIST_INTERVAL_SECONDS = 0.25
HTML_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>DevMate</title>
    <style>
      body {
        font-family: Arial, sans-serif;
        margin: 0;
        background: #f7f7f9;
        color: #1f2937;
      }
      main {
        max-width: 960px;
        margin: 0 auto;
        padding: 32px 20px;
      }
      textarea {
        width: 100%;
        min-height: 140px;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #d1d5db;
        box-sizing: border-box;
      }
      button {
        margin-top: 12px;
        padding: 10px 18px;
        border: 0;
        border-radius: 8px;
        background: #111827;
        color: white;
        cursor: pointer;
      }
      button:disabled {
        opacity: 0.6;
        cursor: wait;
      }
      button.secondary {
        background: #374151;
      }
      input[type="file"] {
        width: 100%;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #d1d5db;
        background: white;
        box-sizing: border-box;
      }
      pre {
        white-space: pre-wrap;
        word-break: break-word;
        background: white;
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
      }
      .panel {
        margin-top: 20px;
      }
      ul {
        background: white;
        padding: 16px 24px;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
      }
      li {
        margin-bottom: 8px;
      }
      .hint {
        color: #6b7280;
      }
      .status {
        margin-top: 12px;
        padding: 12px 14px;
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        border-radius: 8px;
      }
      .status-line + .status-line {
        margin-top: 6px;
      }
      .controls {
        margin-top: 16px;
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        align-items: center;
      }
      .controls label {
        display: inline-flex;
        gap: 8px;
        align-items: center;
      }
      .controls input[type="number"] {
        width: 96px;
        padding: 8px;
        border-radius: 8px;
        border: 1px solid #d1d5db;
      }
      .button-row {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 8px;
      }
      .file-actions {
        margin-left: 8px;
      }
      .file-actions a {
        margin-right: 10px;
      }
      .tag {
        display: inline-block;
        margin-left: 8px;
        padding: 2px 8px;
        font-size: 12px;
        border-radius: 999px;
        background: #e5e7eb;
      }
      .download-link {
        display: inline-block;
        margin-top: 10px;
      }
      .upload-row {
        margin-top: 16px;
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        align-items: center;
      }
      .upload-row button {
        margin-top: 0;
      }
      .upload-meta {
        color: #6b7280;
        font-size: 14px;
      }
    </style>
  </head>
  <body>
    <main>
      <h1>DevMate</h1>
      <p class="hint">
        输入一个项目需求，DevMate 会按 Planner → Researcher → Builder →
        Verifier → Packager 阶段执行，并可选开启“深度优化”闭环。
      </p>
      <textarea id="prompt">我想构建一个展示附近徒步路线的网站项目。</textarea>
      <div class="controls">
        <label>
          <input type="checkbox" id="deepOptimization" />
          开启深度优化
        </label>
        <label>
          <input type="checkbox" id="localPythonToolEnabled" />
          开启本地 Python 调用工具（默认关闭）
        </label>
        <label>
          最大深度优化次数
          <input type="number" id="maxDeepOptimizationRounds" min="0" value="2" />
        </label>
        <label>
          <input type="checkbox" id="includeInitialFiles" />
          本次带入已上传初始文件
        </label>
      </div>
      <section class="panel">
        <h2>初始文件</h2>
        <input type="file" id="initialFileInput" multiple />
        <div class="upload-row">
          <button id="uploadInitialFiles" class="secondary">上传初始文件到 Data</button>
          <span class="upload-meta" id="uploadStatus">尚未上传文件。</span>
        </div>
        <ul id="uploadedInitialFiles"></ul>
      </section>
      <div class="button-row">
        <button id="submit">开始生成</button>
        <button id="finishNow" class="secondary" disabled>立即输出</button>
      </div>
      <div class="status">
        <div class="status-line">
          <strong>当前状态：</strong>
          <span id="statusText">等待开始…</span>
        </div>
        <div class="status-line">
          <strong>当前阶段：</strong>
          <span id="phaseText">尚未进入阶段</span>
        </div>
      </div>
      <section class="panel">
        <h2>Agent 回复</h2>
        <pre id="reply">等待执行…</pre>
      </section>
      <section class="panel">
        <h2>交付信息</h2>
        <pre id="deliveryMeta">等待执行…</pre>
        <a id="deliveryZipLink" class="download-link" href="#" hidden>下载结果压缩包</a>
      </section>
      <section class="panel">
        <h2>变更文件</h2>
        <ul id="changed-files"></ul>
      </section>
      <section class="panel">
        <h2>结果文件</h2>
        <ul id="result-files"></ul>
      </section>
      <section class="panel">
        <h2>文件预览</h2>
        <p class="hint" id="previewTitle">点击结果文件中的“预览”查看内容。</p>
        <pre id="preview">暂无文件预览。</pre>
      </section>
    </main>
    <script>
      const submitButton = document.getElementById('submit');
      const finishNowButton = document.getElementById('finishNow');
      const reply = document.getElementById('reply');
      const statusText = document.getElementById('statusText');
      const phaseText = document.getElementById('phaseText');
      const changedFiles = document.getElementById('changed-files');
      const resultFiles = document.getElementById('result-files');
      const preview = document.getElementById('preview');
      const previewTitle = document.getElementById('previewTitle');
      const deliveryMeta = document.getElementById('deliveryMeta');
      const deliveryZipLink = document.getElementById('deliveryZipLink');
      const includeInitialFiles = document.getElementById('includeInitialFiles');
      const localPythonToolEnabled = document.getElementById('localPythonToolEnabled');
      const initialFileInput = document.getElementById('initialFileInput');
      const uploadInitialFilesButton = document.getElementById('uploadInitialFiles');
      const uploadedInitialFilesList = document.getElementById('uploadedInitialFiles');
      const uploadStatus = document.getElementById('uploadStatus');
      const storageKey = 'devmate.web.currentJobId';
      const POLL_INTERVAL_MS = 350;

      let currentJobId = null;
      let pollHandle = null;
      let pollAbortController = null;
      let pollInFlight = false;
      let polledJobId = null;
      let uploadedInitialFiles = [];

      function encodePath(path) {
        return path.split('/').map(encodeURIComponent).join('/');
      }

      function rememberJob(jobId) {
        if (!jobId) {
          return;
        }
        window.localStorage.setItem(storageKey, jobId);
      }

      function forgetJob() {
        window.localStorage.removeItem(storageKey);
      }

      function stopPolling() {
        if (pollHandle) {
          clearTimeout(pollHandle);
          pollHandle = null;
        }
        if (pollAbortController) {
          pollAbortController.abort();
          pollAbortController = null;
        }
        pollInFlight = false;
        polledJobId = null;
      }

      function shouldKeepPolling(status) {
        return ['queued', 'running', 'interrupted'].includes(status);
      }

      async function readJsonResponse(response) {
        const text = await response.text();
        if (!text) {
          return {};
        }

        try {
          return JSON.parse(text);
        } catch (error) {
          if (response.ok) {
            throw new Error(text || '服务器返回了无法解析的响应');
          }
          return { detail: text };
        }
      }

      function scheduleNextPoll(jobId, delayMs = POLL_INTERVAL_MS) {
        if (polledJobId !== jobId) {
          return;
        }
        pollHandle = window.setTimeout(() => {
          void runPoll(jobId);
        }, delayMs);
      }

      async function runPoll(jobId) {
        if (pollInFlight || polledJobId !== jobId) {
          return;
        }

        pollInFlight = true;
        pollAbortController = new AbortController();
        try {
          const data = await refreshJob(jobId, {
            signal: pollAbortController.signal,
          });
          if (shouldKeepPolling(data.status)) {
            scheduleNextPoll(jobId);
          } else {
            stopPolling();
          }
        } catch (error) {
          if (error.name === 'AbortError') {
            return;
          }
          stopPolling();
          submitButton.disabled = false;
          setFinishButtonState(false, false);
          statusText.textContent = '获取进度失败。';
          reply.textContent = error.message || '获取进度失败';
        } finally {
          pollInFlight = false;
          pollAbortController = null;
        }
      }

      function startPolling(jobId) {
        stopPolling();
        polledJobId = jobId;
        scheduleNextPoll(jobId, 0);
      }

      function setListFallback(element, text) {
        element.innerHTML = '';
        const item = document.createElement('li');
        item.textContent = text;
        element.appendChild(item);
      }

      function setFinishButtonState(isRunning, requested) {
        finishNowButton.disabled = !isRunning || requested;
        finishNowButton.textContent = requested ? '已请求立即输出' : '立即输出';
      }

      function renderUploadedInitialFiles(files) {
        uploadedInitialFiles = files || [];
        uploadedInitialFilesList.innerHTML = '';
        if (!uploadedInitialFiles.length) {
          setListFallback(uploadedInitialFilesList, '暂无已上传初始文件。');
          uploadStatus.textContent = '尚未上传文件。';
          return;
        }

        for (const item of uploadedInitialFiles) {
          const li = document.createElement('li');
          li.textContent = `${item.path} (${item.size_bytes} bytes)`;
          uploadedInitialFilesList.appendChild(li);
        }
        uploadStatus.textContent = (
          `当前已上传 ${uploadedInitialFiles.length} 个初始文件。`
        );
      }

      async function refreshUploadedInitialFiles() {
        const response = await fetch('/api/uploads', {
          cache: 'no-store',
        });
        const data = await readJsonResponse(response);
        if (!response.ok) {
          throw new Error(data.detail || '获取初始文件列表失败');
        }
        renderUploadedInitialFiles(data.files || []);
        return data;
      }

      async function previewFile(path) {
        previewTitle.textContent = path;
        preview.textContent = '正在加载文件内容…';
        try {
          const response = await fetch(`/api/file/${encodePath(path)}`);
          const text = await response.text();
          if (!response.ok) {
            throw new Error(text || '读取文件失败');
          }
          preview.textContent = text;
        } catch (error) {
          preview.textContent = error.message || '读取文件失败';
        }
      }

      function renderChangedFiles(records, savedSkill) {
        changedFiles.innerHTML = '';
        if (!records.length && !savedSkill) {
          setListFallback(changedFiles, '本次执行尚未记录文件操作。');
          return;
        }

        for (const record of records) {
          const item = document.createElement('li');
          item.textContent = record;
          changedFiles.appendChild(item);
        }

        if (savedSkill) {
          const item = document.createElement('li');
          item.textContent = `保存 Skill: ${savedSkill}`;
          changedFiles.appendChild(item);
        }
      }

      function renderResultFiles(files, changedList) {
        resultFiles.innerHTML = '';
        if (!files.length) {
          setListFallback(resultFiles, '暂无可展示的结果文件。');
          return;
        }

        const changedSet = new Set(changedList);
        for (const path of files) {
          const item = document.createElement('li');
          const name = document.createElement('span');
          name.textContent = path;
          item.appendChild(name);

          if (changedSet.has(path)) {
            const tag = document.createElement('span');
            tag.className = 'tag';
            tag.textContent = '新增或修改';
            item.appendChild(tag);
          }

          const actions = document.createElement('span');
          actions.className = 'file-actions';

          const previewLink = document.createElement('a');
          previewLink.href = '#';
          previewLink.textContent = '预览';
          previewLink.addEventListener('click', async (event) => {
            event.preventDefault();
            await previewFile(path);
          });
          actions.appendChild(previewLink);

          const downloadLink = document.createElement('a');
          downloadLink.href = `/api/download/${encodePath(path)}`;
          downloadLink.textContent = '下载';
          downloadLink.target = '_blank';
          downloadLink.rel = 'noopener';
          actions.appendChild(downloadLink);

          item.appendChild(actions);
          resultFiles.appendChild(item);
        }
      }

      function renderDeliveryInfo(data) {
        const verificationText = data.verification_passed === true
          ? '全部细节已完成通过'
          : data.verification_passed === false
            ? '仍有错误或细节未完全实现'
            : '尚未给出结论';
        const initialFileText = data.include_initial_files
          ? `已带入 ${((data.initial_file_paths || []).length)} 个初始文件`
          : '未带入初始文件';
        deliveryMeta.textContent = [
          `深度优化：${data.deep_optimization ? '开启' : '关闭'}`,
          `本地 Python 工具：${data.local_python_tool_enabled ? '开启' : '关闭'}`,
          `最大深度优化次数：${data.max_deep_optimization_rounds ?? 0}`,
          `已使用深度优化次数：${data.optimization_rounds_used ?? 0}`,
          `检验结论：${verificationText}`,
          `初始文件：${initialFileText}`,
          `立即输出：${data.immediate_output_requested ? '已请求' : '未请求'}`,
        ].join('\\n');

        if (data.delivery_zip) {
          deliveryZipLink.hidden = false;
          deliveryZipLink.href = `/api/artifact/${encodePath(data.delivery_zip)}`;
        } else {
          deliveryZipLink.hidden = true;
          deliveryZipLink.href = '#';
        }
      }

      async function refreshJob(jobId, options = {}) {
        const response = await fetch(`/api/jobs/${jobId}`, {
          cache: 'no-store',
          signal: options.signal,
        });
        const data = await readJsonResponse(response);
        if (!response.ok) {
          throw new Error(data.detail || '获取任务状态失败');
        }

        currentJobId = jobId;
        rememberJob(jobId);
        statusText.textContent = data.stage || '任务正在执行…';
        phaseText.textContent = data.current_phase || '尚未进入阶段';
        includeInitialFiles.checked = data.include_initial_files === true;
        localPythonToolEnabled.checked = data.local_python_tool_enabled === true;
        if (data.reply) {
          reply.textContent = data.reply;
        } else if (data.status === 'running' || data.status === 'queued') {
          reply.textContent = '正在等待模型返回最新内容…';
        } else if (data.error) {
          reply.textContent = data.error;
        }

        renderChangedFiles(data.file_operations || [], data.saved_skill);
        renderResultFiles(data.output_files || [], data.changed_files || []);
        renderDeliveryInfo(data);
        setFinishButtonState(
          data.status === 'running' || data.status === 'queued',
          data.immediate_output_requested === true,
        );

        if (data.status === 'completed') {
          stopPolling();
          submitButton.disabled = false;
          setFinishButtonState(false, false);
          statusText.textContent = '任务执行完成。';
          phaseText.textContent = data.current_phase || 'Packager';
          reply.textContent = data.reply || '任务已完成，但未捕获到文本回复。';
        } else if (data.status === 'error') {
          stopPolling();
          submitButton.disabled = false;
          setFinishButtonState(false, false);
          statusText.textContent = '任务执行失败。';
          reply.textContent = data.error || '任务执行失败。';
        } else if (data.status === 'interrupted') {
          submitButton.disabled = false;
          setFinishButtonState(false, data.immediate_output_requested === true);
          statusText.textContent = data.stage || '已恢复最近状态。';
          reply.textContent = data.reply || data.error ||
            '已恢复最近状态，但后台任务未继续运行。';
        }

        return data;
      }

      async function restoreLatestJob() {
        const savedJobId = window.localStorage.getItem(storageKey);
        if (!savedJobId) {
          return;
        }

        currentJobId = savedJobId;
        submitButton.disabled = true;
        statusText.textContent = '正在恢复最近任务状态…';
        try {
          const data = await refreshJob(savedJobId);
          if (shouldKeepPolling(data.status)) {
            startPolling(savedJobId);
          } else {
            submitButton.disabled = false;
          }
        } catch (error) {
          forgetJob();
          submitButton.disabled = false;
          statusText.textContent = '未找到最近任务。';
          reply.textContent = error.message || '恢复任务状态失败';
        }
      }

      uploadInitialFilesButton.addEventListener('click', async () => {
        const files = Array.from(initialFileInput.files || []);
        if (!files.length) {
          uploadStatus.textContent = '请先选择要上传的文件。';
          return;
        }

        uploadInitialFilesButton.disabled = true;
        uploadStatus.textContent = '正在上传初始文件…';
        try {
          const formData = new FormData();
          for (const file of files) {
            formData.append('files', file, file.name);
          }
          const response = await fetch('/api/uploads', {
            method: 'POST',
            body: formData,
          });
          const data = await readJsonResponse(response);
          if (!response.ok) {
            throw new Error(data.detail || '上传初始文件失败');
          }
          renderUploadedInitialFiles(data.files || []);
          initialFileInput.value = '';
          uploadStatus.textContent = '初始文件已上传到 workspace/data/uploads/web。';
        } catch (error) {
          uploadStatus.textContent = error.message || '上传初始文件失败';
        } finally {
          uploadInitialFilesButton.disabled = false;
        }
      });

      submitButton.addEventListener('click', async () => {
        stopPolling();
        forgetJob();
        currentJobId = null;
        submitButton.disabled = true;
        setFinishButtonState(false, false);
        statusText.textContent = '正在创建任务…';
        phaseText.textContent = '尚未进入阶段';
        reply.textContent = '正在执行，请稍候…';
        changedFiles.innerHTML = '';
        resultFiles.innerHTML = '';
        previewTitle.textContent = '点击结果文件中的“预览”查看内容。';
        preview.textContent = '暂无文件预览。';
        deliveryMeta.textContent = '等待执行…';
        deliveryZipLink.hidden = true;
        deliveryZipLink.href = '#';

        try {
          const prompt = document.getElementById('prompt').value;
          const deepOptimization = document.getElementById('deepOptimization').checked;
          const useLocalPythonTool = localPythonToolEnabled.checked;
          const maxRounds = Number(
            document.getElementById('maxDeepOptimizationRounds').value || 0
          );
          const selectedInitialPaths = includeInitialFiles.checked
            ? uploadedInitialFiles.map((item) => item.path)
            : [];
          const response = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              prompt,
              deep_optimization: deepOptimization,
              local_python_tool_enabled: useLocalPythonTool,
              max_deep_optimization_rounds: maxRounds,
              include_initial_files: includeInitialFiles.checked,
              initial_file_paths: selectedInitialPaths,
            })
          });
          const data = await readJsonResponse(response);
          if (!response.ok) {
            throw new Error(data.detail || '任务创建失败');
          }

          currentJobId = data.job_id;
          rememberJob(currentJobId);
          statusText.textContent = '任务已创建，正在获取最新进度…';
          setFinishButtonState(true, false);
          await refreshJob(currentJobId);
          startPolling(currentJobId);
        } catch (error) {
          submitButton.disabled = false;
          setFinishButtonState(false, false);
          statusText.textContent = '任务启动失败。';
          reply.textContent = error.message || '请求失败';
        }
      });

      finishNowButton.addEventListener('click', async () => {
        if (!currentJobId || finishNowButton.disabled) {
          return;
        }
        finishNowButton.disabled = true;
        finishNowButton.textContent = '已请求立即输出';
        statusText.textContent = '已发送立即输出请求，正在切换到最终整合与打包…';
        try {
          const response = await fetch(`/api/jobs/${currentJobId}/immediate-output`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
          });
          const data = await readJsonResponse(response);
          if (!response.ok) {
            throw new Error(data.detail || '发送立即输出请求失败');
          }
          await refreshJob(currentJobId);
        } catch (error) {
          finishNowButton.disabled = false;
          finishNowButton.textContent = '立即输出';
          reply.textContent = error.message || '发送立即输出请求失败';
        }
      });

      setListFallback(uploadedInitialFilesList, '暂无已上传初始文件。');
      setListFallback(changedFiles, '等待执行后展示文件操作记录。');
      setListFallback(resultFiles, '等待执行后展示结果文件。');
      setFinishButtonState(false, false);
      Promise.all([
        refreshUploadedInitialFiles(),
        restoreLatestJob(),
      ]).catch((error) => {
        submitButton.disabled = false;
        statusText.textContent = '初始化页面失败。';
        reply.textContent = error.message || '初始化页面失败';
      });
    </script>
  </body>
</html>
""".strip()


def _json_safe(value: object) -> object:
    """Convert nested values into JSON-serializable primitives."""
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return value.as_posix()

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    return str(value)


def _as_str_list(value: object) -> list[str]:
    """Normalize an arbitrary value into a list of strings."""
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]

    return [str(value)]


def _web_task_is_active(task: asyncio.Task[object] | None) -> bool:
    """Return whether an in-memory Web background task is still active."""

    if task is None:
        return False
    return not task.done() and not task.cancelled()


def _interrupt_web_job_payload(payload: dict[str, object]) -> dict[str, object]:
    """Mark a Web payload as interrupted after confirming it is inactive."""

    normalized = dict(payload)
    normalized["status"] = "interrupted"
    normalized["stage"] = "Web 服务已恢复最近状态，但原后台任务未继续执行。"
    normalized["error"] = normalized.get("error") or (
        "请按当前状态查看结果，必要时重新执行。"
    )
    return normalized


def _reset_web_job_snapshots(app: FastAPI) -> None:
    """Clear stale Web snapshots before a new job starts."""

    active_tasks: dict[str, asyncio.Task[object]] = app.state.active_tasks
    for task in list(active_tasks.values()):
        if not task.done():
            task.cancel()
    active_tasks.clear()
    app.state.jobs.clear()
    runtime: DevMateRuntime = app.state.runtime
    runtime.state_store.clear_scope("web")


async def _run_generation_job(
    runtime: DevMateRuntime,
    job_id: str,
    request_payload: GenerateRequest,
    job_store: dict[str, "GenerationJob"] | None = None,
) -> None:
    """Run a Web generation job in the background."""
    controller = RunController(
        external_request_reader=lambda: runtime.state_store.read_control(
            "web",
            job_id,
        )
    )
    progress_callback = _build_web_progress_callback(
        runtime=runtime,
        job_id=job_id,
        request_payload=request_payload,
        job_store=job_store,
    )

    payload = runtime.state_store.read_state("web", job_id)
    if payload is None:
        payload = _build_web_job_payload(job_id, request_payload)

    payload["status"] = "running"
    payload["stage"] = "正在准备运行环境…"
    payload["error"] = None
    _persist_web_job_payload(runtime, job_id, payload)
    _sync_web_job_store(job_store, job_id, payload)

    try:
        result = await runtime.run(
            request_payload.prompt,
            progress_callback=progress_callback,
            deep_optimization=request_payload.deep_optimization,
            local_python_tool_enabled=(request_payload.local_python_tool_enabled),
            max_deep_optimization_rounds=(request_payload.max_deep_optimization_rounds),
            include_initial_files=request_payload.include_initial_files,
            initial_file_paths=request_payload.initial_file_paths,
            controller=controller,
        )

        payload = runtime.state_store.read_state("web", job_id)
        if payload is None:
            payload = _build_web_job_payload(job_id, request_payload)

        if hasattr(result, "model_dump"):
            result_payload = result.model_dump()
        elif hasattr(result, "__dataclass_fields__"):
            result_payload = asdict(result)
        elif isinstance(result, dict):
            result_payload = result
        else:
            result_payload = {"reply": str(result)}

        payload.update(_json_safe(result_payload))
        payload["status"] = "completed"
        payload["stage"] = "任务执行完成。"
        payload["current_phase"] = "Packager"
        payload["error"] = None
        payload["immediate_output_requested"] = controller.immediate_output_requested

        _persist_web_job_payload(runtime, job_id, payload)
        _sync_web_job_store(job_store, job_id, payload)

    except Exception as exc:
        LOGGER.exception("Generation job %s failed", job_id)
        _persist_web_job_error(
            runtime=runtime,
            job_id=job_id,
            request_payload=request_payload,
            error=exc,
            job_store=job_store,
        )
    finally:
        runtime.state_store.clear_control("web", job_id)


def _build_web_job_payload(
    job_id: str,
    request_payload: GenerateRequest,
) -> dict[str, object]:
    """Create the base payload stored for a web generation job."""
    return {
        "job_id": job_id,
        "prompt": request_payload.prompt,
        "status": "queued",
        "stage": "",
        "current_phase": "",
        "reply": "",
        "changed_files": [],
        "output_files": [],
        "file_operations": [],
        "saved_skill": None,
        "error": None,
        "delivery_zip": None,
        "verification_passed": None,
        "optimization_rounds_used": 0,
        "deep_optimization": request_payload.deep_optimization,
        "local_python_tool_enabled": (request_payload.local_python_tool_enabled),
        "max_deep_optimization_rounds": (request_payload.max_deep_optimization_rounds),
        "include_initial_files": request_payload.include_initial_files,
        "initial_file_paths": list(request_payload.initial_file_paths),
        "immediate_output_requested": False,
        "updated_at": time.time(),
    }


def _persist_web_job_payload(
    runtime: DevMateRuntime,
    job_id: str,
    payload: dict[str, object],
) -> None:
    """Persist a Web job payload snapshot."""
    payload["updated_at"] = time.time()
    runtime.state_store.write_state(
        "web",
        job_id,
        _json_safe(payload),
    )


def _sync_web_job_store(
    job_store: dict[str, "GenerationJob"] | None,
    job_id: str,
    payload: dict[str, object],
) -> None:
    """Mirror the latest payload into the in-memory Web job registry."""

    if job_store is None:
        return

    job_store[job_id] = GenerationJob.from_record(_json_safe(payload))


def _validation_error_uses_invalid_json(exc: RequestValidationError) -> bool:
    """Return whether the validation error was caused by invalid JSON."""

    return any(item.get("type") == "json_invalid" for item in exc.errors())


@dataclass(slots=True)
class WebProgressRecorder:
    """Coalesce high-frequency Web progress updates for smoother UI polling."""

    runtime: DevMateRuntime
    job_id: str
    request_payload: GenerateRequest
    job_store: dict[str, "GenerationJob"] | None = None
    payload: dict[str, object] = field(init=False)
    last_persist_at: float = 0.0
    last_reply_log_at: float = 0.0
    last_reply_log_length: int = 0

    def __post_init__(self) -> None:
        payload = self.runtime.state_store.read_state("web", self.job_id)
        if payload is None:
            payload = _build_web_job_payload(self.job_id, self.request_payload)
        self.payload = dict(payload)

    def _should_log_reply(self, message: str) -> bool:
        """Return whether the current reply update is worth logging."""

        now = time.monotonic()
        message_length = len(message)
        should_log = (
            now - self.last_reply_log_at >= _WEB_REPLY_LOG_INTERVAL_SECONDS
            or message_length - self.last_reply_log_length
            >= _WEB_REPLY_LOG_MIN_LENGTH_DELTA
        )
        if should_log:
            self.last_reply_log_at = now
            self.last_reply_log_length = message_length
        return should_log

    def _log_progress(self, event_type: str, current_phase: str, message: str) -> None:
        """Emit compact progress logs without flooding the server output."""

        if event_type == "reply" and message and not self._should_log_reply(message):
            return

        LOGGER.info(
            "Web job progress | job_id=%s | type=%s | phase=%s | "
            "has_message=%s | message_length=%s",
            self.job_id,
            event_type,
            current_phase,
            bool(message),
            len(message),
        )

    def _should_persist(self, event_type: str) -> bool:
        """Return whether the updated payload should be flushed to disk now."""

        if event_type != "reply":
            return True
        now = time.monotonic()
        if now - self.last_persist_at >= _WEB_REPLY_PERSIST_INTERVAL_SECONDS:
            self.last_persist_at = now
            return True
        return False

    def _apply_event(self, event: dict[str, object]) -> None:
        """Mutate the cached payload using an incoming progress event."""

        payload = self.payload
        event_type = str(event.get("type", "") or "")
        message = str(event.get("message", "") or "")
        current_phase = str(event.get("current_phase", "") or "")

        self._log_progress(event_type, current_phase, message)

        if payload.get("status") not in {"completed", "error"}:
            payload["status"] = "running"

        payload["error"] = None

        if current_phase:
            payload["current_phase"] = current_phase

        if event_type == "status" and message:
            payload["stage"] = message

        if event_type == "reply" and message:
            payload["reply"] = message
            payload["stage"] = "正在接收大模型最新输出…"

        if event_type == "workspace_changes" and message:
            payload["stage"] = message

        if event_type == "final":
            if message:
                payload["stage"] = message
            if event.get("reply"):
                payload["reply"] = str(event["reply"])

        if "changed_files" in event:
            payload["changed_files"] = _as_str_list(event.get("changed_files"))

        if "output_files" in event:
            payload["output_files"] = _as_str_list(event.get("output_files"))

        if "file_operations" in event:
            payload["file_operations"] = _as_str_list(event.get("file_operations"))

        if "saved_skill" in event:
            payload["saved_skill"] = _json_safe(event.get("saved_skill"))

        if "delivery_zip" in event:
            payload["delivery_zip"] = _json_safe(event.get("delivery_zip"))

        if "verification_passed" in event:
            payload["verification_passed"] = _json_safe(
                event.get("verification_passed")
            )

        if "optimization_rounds_used" in event:
            payload["optimization_rounds_used"] = _json_safe(
                event.get("optimization_rounds_used")
            )

        if "immediate_output_requested" in event:
            payload["immediate_output_requested"] = _json_safe(
                event.get("immediate_output_requested")
            )

        payload["updated_at"] = time.time()
        _sync_web_job_store(self.job_store, self.job_id, payload)

        if self._should_persist(event_type):
            _persist_web_job_payload(self.runtime, self.job_id, payload)

    async def __call__(self, event: dict[str, object]) -> None:
        """Apply a progress event to the cached and persisted Web job state."""

        self._apply_event(event)


def _persist_web_job_error(
    runtime: DevMateRuntime,
    job_id: str,
    request_payload: GenerateRequest,
    error: Exception,
    job_store: dict[str, "GenerationJob"] | None = None,
) -> None:
    """Persist a failed Web job so the UI does not hang forever."""
    payload = runtime.state_store.read_state("web", job_id)

    if payload is None:
        payload = _build_web_job_payload(job_id, request_payload)

    payload.update(
        {
            "status": "error",
            "stage": "任务执行失败。",
            "error": str(error),
        }
    )

    _persist_web_job_payload(runtime, job_id, payload)
    _sync_web_job_store(job_store, job_id, payload)


def _build_web_progress_callback(
    runtime: DevMateRuntime,
    job_id: str,
    request_payload: GenerateRequest,
    job_store: dict[str, "GenerationJob"] | None = None,
):
    """Build a JSON-safe progress callback for background Web jobs."""

    return WebProgressRecorder(
        runtime=runtime,
        job_id=job_id,
        request_payload=request_payload,
        job_store=job_store,
    )


@dataclass(slots=True)
class GenerationJob:
    """In-memory state for a background generation job."""

    job_id: str
    prompt: str
    deep_optimization: bool = False
    local_python_tool_enabled: bool = False
    max_deep_optimization_rounds: int = 0
    include_initial_files: bool = False
    initial_file_paths: list[str] = field(default_factory=list)
    status: str = "queued"
    stage: str = "任务已创建，等待执行…"
    current_phase: str = "尚未进入阶段"
    reply: str = ""
    changed_files: list[str] = field(default_factory=list)
    output_files: list[str] = field(default_factory=list)
    file_operations: list[str] = field(default_factory=list)
    saved_skill: str | None = None
    error: str | None = None
    delivery_zip: str | None = None
    verification_passed: bool | None = None
    optimization_rounds_used: int = 0
    updated_at: float = field(default_factory=time.time)
    controller: RunController = field(default_factory=RunController)

    def to_response(self) -> GenerationJobStatusResponse:
        """Convert the internal job state to an API schema."""

        return GenerationJobStatusResponse(
            job_id=self.job_id,
            prompt=self.prompt,
            status=self.status,
            stage=self.stage,
            current_phase=self.current_phase,
            reply=self.reply,
            changed_files=self.changed_files,
            output_files=self.output_files,
            file_operations=self.file_operations,
            saved_skill=self.saved_skill,
            error=self.error,
            delivery_zip=self.delivery_zip,
            verification_passed=self.verification_passed,
            optimization_rounds_used=self.optimization_rounds_used,
            deep_optimization=self.deep_optimization,
            local_python_tool_enabled=self.local_python_tool_enabled,
            max_deep_optimization_rounds=self.max_deep_optimization_rounds,
            include_initial_files=self.include_initial_files,
            initial_file_paths=self.initial_file_paths,
            immediate_output_requested=self.controller.immediate_output_requested,
        )

    def to_record(self) -> dict[str, object]:
        """Serialize the job state for persistent storage."""

        payload = self.to_response().model_dump()
        payload["updated_at"] = self.updated_at
        return payload

    @classmethod
    def from_record(cls, payload: dict[str, object]) -> GenerationJob:
        """Rebuild a persisted job state from JSON storage."""

        controller = RunController()
        if bool(payload.get("immediate_output_requested", False)):
            controller.request_immediate_output(
                str(payload.get("stage", "恢复时已存在立即输出请求"))
            )

        verification_passed_value = payload.get("verification_passed")
        verification_passed = (
            bool(verification_passed_value)
            if verification_passed_value is not None
            else None
        )
        job = cls(
            job_id=str(payload.get("job_id", "")),
            prompt=str(payload.get("prompt", "")),
            deep_optimization=bool(payload.get("deep_optimization", False)),
            local_python_tool_enabled=bool(
                payload.get("local_python_tool_enabled", False)
            ),
            max_deep_optimization_rounds=int(
                payload.get("max_deep_optimization_rounds", 0)
            ),
            include_initial_files=bool(payload.get("include_initial_files", False)),
            initial_file_paths=list(payload.get("initial_file_paths", [])),
            status=str(payload.get("status", "queued")),
            stage=str(payload.get("stage", "任务已创建，等待执行…")),
            current_phase=str(payload.get("current_phase", "尚未进入阶段")),
            reply=str(payload.get("reply", "")),
            changed_files=list(payload.get("changed_files", [])),
            output_files=list(payload.get("output_files", [])),
            file_operations=list(payload.get("file_operations", [])),
            saved_skill=(
                str(payload.get("saved_skill")) if payload.get("saved_skill") else None
            ),
            error=str(payload.get("error")) if payload.get("error") else None,
            delivery_zip=(
                str(payload.get("delivery_zip"))
                if payload.get("delivery_zip")
                else None
            ),
            verification_passed=verification_passed,
            optimization_rounds_used=int(payload.get("optimization_rounds_used", 0)),
            updated_at=float(payload.get("updated_at", time.time())),
            controller=controller,
        )
        return job


def _persist_generation_job_state(
    runtime: DevMateRuntime,
    job: GenerationJob,
) -> None:
    """Write the latest GenerationJob state to the persistent store."""
    runtime.state_store.write_state("web", job.job_id, job.to_record())


def _load_web_job(
    runtime: DevMateRuntime,
    job_id: str,
    active_tasks: dict[str, asyncio.Task[object]] | None = None,
    job_store: dict[str, GenerationJob] | None = None,
) -> GenerationJob | None:
    """Load a persisted Web job if present."""

    if job_store is not None and job_id in job_store:
        job = job_store[job_id]
        task = None if active_tasks is None else active_tasks.get(job_id)
        if job.status in {"queued", "running"} and not _web_task_is_active(task):
            record = _interrupt_web_job_payload(job.to_record())
            _persist_web_job_payload(runtime, job_id, record)
            job = GenerationJob.from_record(record)
            job_store[job_id] = job
        return job

    record = runtime.state_store.read_state("web", job_id)
    if record is None:
        return None

    status = str(record.get("status", ""))
    task = None if active_tasks is None else active_tasks.get(job_id)
    if status in {"queued", "running"} and not _web_task_is_active(task):
        record = _interrupt_web_job_payload(record)
        _persist_web_job_payload(runtime, job_id, record)

    _sync_web_job_store(job_store, job_id, record)

    return GenerationJob.from_record(record)


def _load_persisted_web_jobs(runtime: DevMateRuntime) -> dict[str, GenerationJob]:
    """Load persisted Web jobs during application startup."""

    jobs: dict[str, GenerationJob] = {}
    for record in runtime.state_store.list_states("web"):
        job = GenerationJob.from_record(record)
        if job.job_id:
            jobs[job.job_id] = job
    return jobs


def _list_uploaded_initial_files(
    runtime: DevMateRuntime,
    scope: str,
) -> UploadedInitialFileList:
    """Return uploaded initial file metadata for the given scope."""

    files = [
        UploadedInitialFile(**item)
        for item in runtime.describe_initial_upload_files(scope)
    ]
    return UploadedInitialFileList(files=files)


def create_app(config_path: str | None = None) -> FastAPI:
    """Create the FastAPI application."""

    config = load_config(config_path)
    configure_logging(config.app.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = DevMateRuntime(config)
        await runtime.prepare(rebuild_kb=False)
        app.state.runtime = runtime
        app.state.jobs = _load_persisted_web_jobs(runtime)
        yield

    app = FastAPI(title="DevMate", lifespan=lifespan)
    app.state.active_tasks = {}

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        """Serve the DevMate web UI."""

        return HTMLResponse(HTML_TEMPLATE)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Liveness probe."""

        return {"status": "ok"}

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Return stable JSON errors for malformed request bodies."""

        if _validation_error_uses_invalid_json(exc):
            return JSONResponse(
                status_code=400,
                content={
                    "detail": "请求体不是有效的 JSON，请检查前端请求后重试。"
                },
            )

        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors()},
        )

    @app.get("/api/uploads", response_model=UploadedInitialFileList)
    async def list_uploaded_initial_files(request: Request) -> UploadedInitialFileList:
        """List uploaded initial files stored for the Web UI."""

        runtime: DevMateRuntime = request.app.state.runtime
        return _list_uploaded_initial_files(runtime, "web")

    @app.post("/api/uploads", response_model=UploadedInitialFileList)
    async def upload_initial_files(
        request: Request,
        files: Annotated[list[UploadFile], File(...)],
    ) -> UploadedInitialFileList:
        """Upload initial files into the Web upload area."""

        runtime: DevMateRuntime = request.app.state.runtime
        for item in files:
            filename = item.filename or "upload.bin"
            content = await item.read()
            runtime.save_uploaded_file(filename, content, "web")
        return _list_uploaded_initial_files(runtime, "web")

    @app.post("/api/generate", response_model=GenerationJobStartResponse)
    async def start_generate_job(
        payload: GenerateRequest,
        request: Request,
    ) -> GenerationJobStartResponse:
        """Start a background generation job and return its identifier."""

        runtime: DevMateRuntime = request.app.state.runtime
        _reset_web_job_snapshots(request.app)
        job_id = uuid4().hex

        initial_payload = _build_web_job_payload(
            job_id=job_id,
            request_payload=payload,
        )
        initial_payload["status"] = "queued"
        initial_payload["stage"] = "任务已提交，正在启动。"

        _persist_web_job_payload(runtime, job_id, initial_payload)
        request.app.state.jobs[job_id] = GenerationJob.from_record(initial_payload)

        task = asyncio.create_task(
            _run_generation_job(
                runtime=runtime,
                job_id=job_id,
                request_payload=payload,
                job_store=request.app.state.jobs,
            )
        )
        request.app.state.active_tasks[job_id] = task

        def _cleanup_task(_task: asyncio.Task[object]) -> None:
            request.app.state.active_tasks.pop(job_id, None)

        task.add_done_callback(_cleanup_task)

        return GenerationJobStartResponse(
            job_id=job_id,
            status="queued",
        )

    @app.post("/api/generate-sync", response_model=GenerateResponse)
    async def generate_sync(
        payload: GenerateRequest,
        request: Request,
    ) -> GenerateResponse:
        """Run the DevMate agent synchronously against a user prompt."""

        runtime: DevMateRuntime = request.app.state.runtime
        controller = RunController()
        try:
            result = await runtime.run(
                payload.prompt,
                deep_optimization=payload.deep_optimization,
                local_python_tool_enabled=payload.local_python_tool_enabled,
                max_deep_optimization_rounds=payload.max_deep_optimization_rounds,
                include_initial_files=payload.include_initial_files,
                initial_file_paths=payload.initial_file_paths,
                controller=controller,
            )
        except Exception as exc:  # pragma: no cover - surfaced to UI
            raise HTTPException(
                status_code=500,
                detail=str(exc),
            ) from exc
        return GenerateResponse(
            prompt=result.prompt,
            reply=result.reply,
            changed_files=result.changed_files,
            output_files=result.output_files,
            file_operations=result.file_operations,
            saved_skill=result.saved_skill,
            delivery_zip=result.delivery_zip,
            verification_passed=result.verification_passed,
            optimization_rounds_used=result.optimization_rounds_used,
            deep_optimization=result.deep_optimization,
            local_python_tool_enabled=result.local_python_tool_enabled,
            max_deep_optimization_rounds=result.max_deep_optimization_rounds,
            include_initial_files=payload.include_initial_files,
            initial_file_paths=payload.initial_file_paths,
            immediate_output_requested=result.immediate_output_requested,
        )

    @app.get("/api/jobs/{job_id}", response_model=GenerationJobStatusResponse)
    async def get_job_status(
        job_id: str,
        request: Request,
    ) -> GenerationJobStatusResponse:
        """Return the current state of a background generation job."""

        runtime: DevMateRuntime = request.app.state.runtime
        job = _load_web_job(
            runtime,
            job_id,
            request.app.state.active_tasks,
            request.app.state.jobs,
        )

        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        request.app.state.jobs[job_id] = job
        return job.to_response()

    @app.post(
        "/api/jobs/{job_id}/immediate-output",
        response_model=GenerationJobStatusResponse,
    )
    async def request_immediate_output(
        job_id: str,
        request: Request,
    ) -> GenerationJobStatusResponse:
        """Ask a running job to jump to Packager as soon as possible."""

        runtime: DevMateRuntime = request.app.state.runtime
        job = _load_web_job(
            runtime,
            job_id,
            request.app.state.active_tasks,
            request.app.state.jobs,
        )

        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.status not in {"queued", "running"}:
            return job.to_response()

        reason = "用户在 Web 前端点击了立即输出"
        job.controller.request_immediate_output(reason)

        runtime.state_store.write_control(
            "web",
            job.job_id,
            {
                "immediate_output_requested": True,
                "reason": reason,
                "updated_at": time.time(),
            },
        )

        job.updated_at = time.time()
        if job.status == "queued":
            job.stage = "已收到立即输出请求，任务启动后会直接进入最终整合与打包。"
        else:
            job.stage = "已收到立即输出请求，将终止当前阶段并进入最终整合与打包。"

        request.app.state.jobs[job_id] = job
        _persist_generation_job_state(runtime, job)
        return job.to_response()

    @app.get("/api/workspace", response_model=WorkspaceListing)
    async def workspace(request: Request) -> WorkspaceListing:
        """List files under the workspace directory."""

        runtime: DevMateRuntime = request.app.state.runtime
        return WorkspaceListing(files=runtime.list_workspace_files())

    @app.get("/api/file/{relative_path:path}", response_class=PlainTextResponse)
    async def read_workspace_file(
        relative_path: str,
        request: Request,
    ) -> PlainTextResponse:
        """Read a generated workspace file as plain text."""
        runtime: DevMateRuntime = request.app.state.runtime

        try:
            candidate = runtime.resolve_workspace_path(relative_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        try:
            content = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail="File is not valid UTF-8 text",
            ) from exc

        return PlainTextResponse(content)

    @app.get("/api/download/{relative_path:path}")
    async def download_workspace_file(
        relative_path: str,
        request: Request,
    ) -> FileResponse:
        """Download a generated workspace file."""
        runtime: DevMateRuntime = request.app.state.runtime

        try:
            candidate = runtime.resolve_workspace_path(relative_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(path=candidate, filename=candidate.name)

    @app.get("/api/artifact/{relative_path:path}")
    async def download_artifact_file(
        relative_path: str,
        request: Request,
    ) -> FileResponse:
        """Download a workflow artifact such as the packaged zip file."""
        runtime: DevMateRuntime = request.app.state.runtime

        try:
            candidate = runtime.resolve_workflow_artifact_path(relative_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")

        return FileResponse(path=candidate, filename=candidate.name)

    return app
