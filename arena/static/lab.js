/* Dataset collection and behaviour cloning. API values are rendered as text. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const items = value => Array.isArray(value) ? value : [];
  const number = value => Number.isFinite(Number(value)) ? Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 }) : '—';
  const entrants = new Set(), collectionTeachers = new Set(), datasetIds = new Set(), trainingTeachers = new Set();
  const teacherWeights = new Map(), storageOverrides = new Map(), pending = new Set();
  const datasetCards = new Map(), jobCards = new Map(), actionMessages = new Map();
  let state = { agents: [], datasets: [], training_jobs: [], recorded_runs: [] };
  let entrantSignature = '', teacherSignature = '', datasetSignature = '', trainingTeacherSignature = '', importSignature = '';
  let eligibleCollectionTeachers = new Set(), eligibleTrainingTeachers = new Set(), refreshing = false;
  let phhDefaultsSet = false, phhBusy = false;

  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text != null) node.textContent = String(text);
    if (className) node.className = className;
    return node;
  }
  function message(id, text, error = false) {
    $(id).textContent = text;
    $(id).classList.toggle('error-text', error);
  }
  function empty(parent, title, text) {
    const box = element('div', null, 'empty');
    box.append(element('strong', title), element('span', text));
    parent.replaceChildren(box);
  }
  async function api(path, body) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60000);
    try {
      const options = { signal: controller.signal, cache: 'no-store' };
      if (body !== undefined) {
        options.method = 'POST';
        options.headers = { 'Content-Type': 'application/json' };
        options.body = JSON.stringify(body);
      }
      const response = await fetch(path, options);
      const text = await response.text();
      let data;
      try { data = text ? JSON.parse(text) : {}; }
      catch { throw new Error(`The server returned an unreadable response (${response.status}).`); }
      if (!response.ok) throw new Error(data.error || data.detail || `Request failed (${response.status}).`);
      return data;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('The request timed out. Check the job list before retrying; the server may have accepted it.');
      throw error;
    } finally { clearTimeout(timeout); }
  }
  function bytes(value) {
    let amount = Number(value || 0), unit = 0;
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    while (amount >= 1024 && unit < units.length - 1) { amount /= 1024; unit++; }
    return `${number(amount)} ${units[unit]}`;
  }
  function duration(value) {
    if (!Number.isFinite(Number(value)) || Number(value) < 0) return 'Estimating…';
    const seconds = Math.round(Number(value));
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor(seconds % 3600 / 60)}m`;
  }
  function date(value) {
    if (!value) return '—';
    const parsed = new Date(typeof value === 'number' && value < 1e12 ? value * 1000 : value);
    return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString();
  }
  function selectedSizes(name) {
    return Array.from(document.querySelectorAll(`input[name="${name}"]:checked`), input => Number(input.value));
  }
  function integer(id, label, minimum, maximum = Number.MAX_SAFE_INTEGER) {
    const value = Number($(id).value);
    if (!Number.isSafeInteger(value) || value < minimum || value > maximum) throw new Error(`${label} must be a whole number from ${number(minimum)} to ${number(maximum)}.`);
    return value;
  }
  function checkboxRow(item, checked, onChange, detail) {
    const row = element('label', null, 'lab-check-row'), checkbox = element('input');
    checkbox.type = 'checkbox'; checkbox.value = String(item.id); checkbox.checked = checked;
    checkbox.addEventListener('change', () => onChange(checkbox.checked, checkbox.value));
    const text = element('span', null, 'agent-info');
    text.append(element('strong', item.name || item.id));
    if (detail) text.append(element('small', detail));
    row.append(checkbox, text);
    return row;
  }
  function isBaseline(agent) {
    const name = String(agent.name || '').toLowerCase().replace(/[^a-z0-9]/g, '');
    return /alwaysfold|alwaysallin|alwasallin|scaffold/.test(name);
  }
  function updateSet(set, checked, value) { checked ? set.add(value) : set.delete(value); }

  // Keep form selections when the background status poll updates the library.
  function renderEntrants() {
    const agents = items(state.agents);
    const available = new Set(agents.map(agent => String(agent.id)));
    for (const id of entrants) if (!available.has(id)) entrants.delete(id);
    const signature = JSON.stringify(agents.map(agent => [agent.id, agent.name, agent.hash]));
    if (signature !== entrantSignature) {
      entrantSignature = signature;
      $('dataset-agent-list').replaceChildren();
      if (!agents.length) empty($('dataset-agent-list'), 'Register bots first', 'Add your bots in the Model library, then return here to collect their decisions.');
      for (const agent of agents) {
        $('dataset-agent-list').append(checkboxRow(agent, entrants.has(String(agent.id)), (checked, id) => {
          updateSet(entrants, checked, id); renderCollectionTeachers(); updateCollectionEstimate();
        }, `Version ${String(agent.hash || agent.id).slice(0, 12)}`));
      }
    }
    renderCollectionTeachers();
    updateCollectionEstimate();
  }
  function renderCollectionTeachers() {
    const teachers = items(state.agents).filter(agent => entrants.has(String(agent.id)));
    const eligible = new Set(teachers.map(agent => String(agent.id)));
    for (const id of collectionTeachers) if (!eligible.has(id)) collectionTeachers.delete(id);
    for (const teacher of teachers) {
      const id = String(teacher.id);
      if (!eligibleCollectionTeachers.has(id) && !isBaseline(teacher)) collectionTeachers.add(id);
    }
    eligibleCollectionTeachers = eligible;
    const signature = JSON.stringify(teachers.map(agent => [agent.id, agent.name, agent.hash]));
    if (signature === teacherSignature) return;
    teacherSignature = signature;
    $('dataset-teacher-list').replaceChildren();
    if (!teachers.length) empty($('dataset-teacher-list'), 'Select entrants first', 'Selected strategy bots become teachers by default.');
    for (const teacher of teachers) {
      $('dataset-teacher-list').append(checkboxRow(teacher, collectionTeachers.has(String(teacher.id)), (checked, id) => {
        updateSet(collectionTeachers, checked, id); updateCollectionEstimate();
      }, isBaseline(teacher) ? 'Baseline · select explicitly to record' : `Version ${String(teacher.hash || teacher.id).slice(0, 12)}`));
    }
  }
  function collectionPayload() {
    const sizes = selectedSizes('dataset-size');
    if (!sizes.length) throw new Error('Choose at least one table size.');
    if (entrants.size < Math.max(...sizes)) throw new Error(`Select at least ${Math.max(...sizes)} entrants for these table sizes.`);
    if (!collectionTeachers.size) throw new Error('Select at least one teacher to record.');
    const storage = Number($('dataset-storage').value);
    if (!Number.isFinite(storage) || storage < 0.1 || storage > 1000) throw new Error('Storage limit must be between 0.1 and 1,000 GB.');
    const payload = {
      name: $('dataset-name').value.trim(), agents: [...entrants], teachers: [...collectionTeachers], table_sizes: sizes,
      target_episodes: integer('dataset-target', 'Target episodes', 1, 100000000), deals: integer('dataset-deals', 'Hands per game', 1, 1000),
      max_disk_gb: storage
    };
    const prefix = $('dataset-seed').value.trim();
    if (prefix) payload.seed_prefix = prefix;
    return payload;
  }
  function updateCollectionEstimate() {
    try {
      const payload = collectionPayload();
      let games = 0;
      for (const size of payload.table_sizes) games += (entrants.size === size ? 1 : entrants.size) * size;
      const episodesPerBatch = games * payload.deals;
      const batches = Math.max(3, Math.ceil(payload.target_episodes / episodesPerBatch));
      const plannedEpisodes = batches * episodesPerBatch;
      message('dataset-estimate', `${number(entrants.size)} entrants · ${number(collectionTeachers.size)} teachers. Each balanced batch plays ${number(games)} games / ${number(episodesPerBatch)} episodes. Your ${number(payload.target_episodes)}-episode target rounds to ${number(plannedEpisodes)} episodes across ${number(batches)} full batches, assuming valid teacher data. This includes the minimum three seed groups. Training decision counts will depend on play.`);
    } catch (error) { message('dataset-estimate', error.message, true); }
  }
  function renderPHHControls() {
    const available = state.phh?.available === true;
    $('phh-import').hidden = !available;
    $('phh-shortcut').hidden = !available;
    if (available && !phhDefaultsSet) {
      if (state.phh.default_source_path) $('phh-source').value = state.phh.default_source_path;
      phhDefaultsSet = true;
    }
    const cpuOnly = state.development_mode === true || state.phh?.preview_cpu_only === true;
    const device = $('cloning-device');
    if (cpuOnly) device.value = 'cpu';
    device.disabled = cpuOnly;
    device.querySelector('option[value="auto"]').disabled = cpuOnly;
    device.querySelector('option[value="mps"]').disabled = cpuOnly;
    device.querySelector('option[value="cuda"]').disabled = cpuOnly;
    $('cloning-device-note').hidden = !cpuOnly;
  }
  function phhPayload() {
    if (state.phh?.available !== true) throw new Error('PHH import is unavailable in this workspace.');
    const sourcePath = $('phh-source').value.trim();
    if (!sourcePath) throw new Error('Enter the PHH source folder.');
    return {
      source_path: sourcePath,
      collection: $('phh-collection').value,
      teacher_name: $('phh-teacher').value.trim(),
      max_hands: integer('phh-max-hands', 'Maximum hands', 1),
      profitable_only: $('phh-profitable').checked
    };
  }
  function renderPHHPreview(preview) {
    const result = $('phh-preview');
    result.replaceChildren();
    result.append(element('p', `${number(preview.sample_hands)} hands checked · ${number(preview.usable_hands)} usable hands · ${number(preview.records)} training decisions`));
    result.append(element('p', `Teachers: ${items(preview.teachers).join(', ') || 'None in this sample'}. Table sizes: ${items(preview.table_sizes).join(', ') || 'None in this sample'}.`));
    const skipped = Object.entries(preview.skipped || {});
    if (skipped.length) result.append(table(skipped.map(([reason, count]) => [reason.replace(/_/g, ' '), number(count)]), ['Skipped reason', 'Count']));
    if (preview.note) result.append(element('p', preview.note));
    if (preview.source_folder) result.append(element('p', `Source folder: ${preview.source_folder}`, 'lab-path'));
    result.hidden = false;
  }
  async function submitPHH(previewOnly) {
    if (phhBusy) return;
    phhBusy = true;
    $('phh-preview-button').disabled = true;
    $('phh-import-button').disabled = true;
    try {
      const payload = phhPayload();
      const settings = JSON.stringify(payload);
      if (!previewOnly) {
        payload.name = $('phh-name').value.trim();
        if (!payload.name) throw new Error('Enter a dataset name.');
      }
      message('phh-status', previewOnly ? 'Checking up to 100 hands…' : 'Queuing PHH import…');
      const result = await api(previewOnly ? '/api/phh/preview' : '/api/phh/import', payload);
      if (previewOnly) {
        if (settings !== JSON.stringify(phhPayload())) {
          message('phh-status', 'Settings changed while the preview ran. Preview again to check this selection.');
          return;
        }
        renderPHHPreview(result);
        message('phh-status', 'Preview complete. This sample may not represent the full collection.');
      } else {
        message('phh-status', 'PHH import queued. Follow it in the dataset library, then choose “Use for training” when ready.');
        await refresh();
      }
    } catch (error) { message('phh-status', error.message, true); }
    finally {
      phhBusy = false;
      $('phh-preview-button').disabled = false;
      $('phh-import-button').disabled = false;
    }
  }
  function datasetTeachers() {
    const teachers = new Map();
    for (const dataset of items(state.datasets)) {
      if (!datasetIds.has(String(dataset.id))) continue;
      for (const teacher of items(dataset.teachers)) {
        const id = String(teacher.id);
        if (!teachers.has(id)) teachers.set(id, { ...teacher, id, records: 0 });
        teachers.get(id).records += Number(dataset.teacher_counts?.[id] || 0);
      }
    }
    return [...teachers.values()];
  }
  function renderTrainingChoices() {
    const ready = items(state.datasets).filter(dataset => dataset.status === 'ready');
    const available = new Set(ready.map(dataset => String(dataset.id)));
    for (const id of datasetIds) if (!available.has(id)) datasetIds.delete(id);
    const signature = JSON.stringify(ready.map(dataset => [dataset.id, dataset.name, dataset.records]));
    if (signature !== datasetSignature) {
      datasetSignature = signature;
      $('cloning-dataset-list').replaceChildren();
      if (!ready.length) empty($('cloning-dataset-list'), 'Your data comes first', state.phh?.available ? 'Finish a collection, import PHH, or add a completed recorded run.' : 'Finish a collection or import a completed recorded run.');
      for (const dataset of ready) {
        $('cloning-dataset-list').append(checkboxRow(dataset, datasetIds.has(String(dataset.id)), (checked, id) => {
          updateSet(datasetIds, checked, id); renderTrainingTeachers(); updateTrainingEstimate();
        }, `${number(dataset.records)} decisions · ${bytes(dataset.bytes)} · ${date(dataset.created)}`));
      }
    }
    renderTrainingTeachers();
    updateTrainingEstimate();
  }
  function renderTrainingTeachers() {
    const teachers = datasetTeachers();
    const eligible = new Set(teachers.map(teacher => teacher.id));
    for (const id of trainingTeachers) if (!eligible.has(id)) trainingTeachers.delete(id);
    for (const teacher of teachers) {
      if (!eligibleTrainingTeachers.has(teacher.id) && !isBaseline(teacher)) trainingTeachers.add(teacher.id);
    }
    eligibleTrainingTeachers = eligible;
    const signature = JSON.stringify(teachers);
    if (signature === trainingTeacherSignature) return;
    trainingTeacherSignature = signature;
    $('cloning-teacher-list').replaceChildren();
    if (!teachers.length) empty($('cloning-teacher-list'), 'Choose a dataset', 'Its teacher versions will appear here.');
    for (const teacher of teachers) {
      const row = element('div', null, 'lab-weight-row');
      const choice = checkboxRow(teacher, trainingTeachers.has(teacher.id), (checked, id) => {
        updateSet(trainingTeachers, checked, id); updateTrainingEstimate();
      }, `${number(teacher.records)} decisions · ${String(teacher.hash || teacher.id).slice(0, 12)}`);
      const label = element('label', 'Weight'), input = element('input');
      input.type = 'number'; input.min = '0.000001'; input.max = '1000'; input.step = 'any';
      input.value = teacherWeights.get(teacher.id) || '1'; input.setAttribute('aria-label', `Weight for ${teacher.name || teacher.id}`);
      input.addEventListener('input', () => teacherWeights.set(teacher.id, input.value));
      label.append(input); row.append(choice, label); $('cloning-teacher-list').append(row);
    }
  }
  function updateTrainingEstimate() {
    const sizes = selectedSizes('cloning-size');
    const datasets = items(state.datasets).filter(dataset => datasetIds.has(String(dataset.id)));
    if (!datasets.length || !trainingTeachers.size || !sizes.length) {
      message('cloning-estimate', 'Choose a completed dataset, at least one teacher and at least one table size.');
      return;
    }
    let records = 0, exact = true;
    for (const dataset of datasets) {
      if (!dataset.teacher_size_counts) { exact = false; records += Number(dataset.records || 0); continue; }
      for (const teacher of trainingTeachers) for (const size of sizes) records += Number(dataset.teacher_size_counts?.[teacher]?.[size] || 0);
    }
    message('cloning-estimate', `${datasets.length} dataset${datasets.length === 1 ? '' : 's'} · ${trainingTeachers.size} teachers · ${sizes.join(', ')} players. ${exact ? 'Approximately' : 'Up to'} ${number(records)} decisions before duplicate removal and the training/validation/test split.`);
  }
  function trainingPayload() {
    const sizes = selectedSizes('cloning-size');
    if (!datasetIds.size || !trainingTeachers.size || !sizes.length) throw new Error('Choose a dataset, at least one teacher and at least one table size.');
    const learningRate = Number($('cloning-rate').value);
    if (!Number.isFinite(learningRate) || learningRate < 0.000001 || learningRate > 0.1) throw new Error('Learning rate must be between 0.000001 and 0.1.');
    const weights = {};
    for (const id of trainingTeachers) {
      const weight = Number(teacherWeights.get(id) || 1);
      if (!Number.isFinite(weight) || weight <= 0 || weight > 1000) throw new Error('Teacher weights must be greater than 0 and at most 1,000.');
      weights[id] = weight;
    }
    return {
      name: $('cloning-name').value.trim(), dataset_ids: [...datasetIds], teachers: [...trainingTeachers], table_sizes: sizes,
      epochs: integer('cloning-epochs', 'Epochs', 1, 200), batch_size: integer('cloning-batch', 'Batch size', 16, 8192),
      learning_rate: learningRate, balanced: $('cloning-balanced').checked, teacher_weights: weights,
      device: state.development_mode === true || state.phh?.preview_cpu_only === true ? 'cpu' : $('cloning-device').value
    };
  }

  function table(rows, headers) {
    const wrap = element('div', null, 'table-wrap'), result = element('table'), head = element('thead'), heading = element('tr');
    for (const title of headers) { const cell = element('th', title); cell.scope = 'col'; heading.append(cell); }
    head.append(heading); result.append(head);
    const body = element('tbody');
    for (const values of rows) {
      const row = element('tr');
      for (const value of values) row.append(element('td', value == null ? '—' : value));
      body.append(row);
    }
    result.append(body); wrap.append(result); return wrap;
  }
  function progressBar(value, maximum, label) {
    const wrap = element('div', null, 'lab-progress');
    const progress = element('progress'); progress.max = Math.max(1, Number(maximum || 1)); progress.value = Math.min(progress.max, Math.max(0, Number(value || 0)));
    progress.setAttribute('aria-label', label);
    wrap.append(progress, element('span', label, 'small muted')); return wrap;
  }
  function download(path, text) {
    const link = element('a', text, 'button secondary'); link.href = path; link.download = ''; return link;
  }
  function actionButton(kind, id, action, text) {
    const button = element('button', text, 'button secondary'); button.type = 'button';
    button.dataset.labKind = kind; button.dataset.labId = String(id); button.dataset.labAction = action;
    button.disabled = pending.has(`${kind}/${id}`);
    return button;
  }
  function statusLabel(item, kind = 'datasets') {
    if (item.pause_requested && item.status === 'running') return kind === 'training' && item.finish_requested ? 'Pausing finalization…' : 'Pausing…';
    if (item.finish_requested && ['queued', 'running'].includes(item.status)) {
      if (kind === 'training') return item.status === 'queued' ? 'Queued to finalize best checkpoint' : 'Finalizing best checkpoint';
      return item.kind === 'phh' ? 'Finishing import' : 'Finishing batch';
    }
    return String(item.status || 'unknown').replace(/_/g, ' ');
  }
  function progressText(item) {
    if (typeof item.progress === 'string') return item.progress;
    if (item.progress?.message) return item.progress.message;
    if (typeof item.progress?.progress === 'string') return item.progress.progress;
    return '';
  }
  function updateCards(parentId, cache, records, render, emptyTitle, emptyText) {
    const parent = $(parentId), ids = new Set(records.map(record => String(record.id)));
    for (const [id, card] of cache) if (!ids.has(id)) { card.node.remove(); cache.delete(id); }
    if (!records.length) { empty(parent, emptyTitle, emptyText); return; }
    if (parent.querySelector(':scope > .empty')) parent.replaceChildren();
    for (const record of records) {
      const id = String(record.id), key = `${parentId === 'dataset-library' ? 'datasets' : 'training'}/${id}`;
      const signature = JSON.stringify([record, pending.has(key), actionMessages.get(key)]);
      let card = cache.get(id);
      if (!card) {
        card = { node: element('article', null, 'lab-card'), signature: '' };
        cache.set(id, card); parent.append(card.node);
      }
      if (card.signature === signature) continue;
      const openDetails = [...card.node.querySelectorAll('details')].map(details => details.open);
      render(card.node, record);
      const actionStatus = actionMessages.get(key), status = element('div', actionStatus?.text || '', 'small lab-action-status');
      status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite');
      if (actionStatus?.error) status.classList.add('error-text');
      status.hidden = !actionStatus; card.node.append(status);
      card.node.querySelectorAll('details').forEach((details, index) => { details.open = openDetails[index] || false; });
      card.signature = signature;
    }
    records.forEach((record, index) => {
      const node = cache.get(String(record.id)).node;
      if (parent.children[index] !== node) parent.insertBefore(node, parent.children[index] || null);
    });
  }
  function cardHeading(node, item, subtitle, kind = 'datasets') {
    const heading = element('div', null, 'lab-card-heading'), title = element('div', null, 'run-title');
    title.append(element('strong', item.name || item.id), element('small', subtitle));
    const tag = element('span', statusLabel(item, kind), 'tag');
    if (/^(ready|completed|running|queued|paused|interrupted|failed)$/.test(item.status || '')) tag.classList.add(`status-${item.status}`);
    heading.append(title, tag); node.append(heading);
  }
  function renderDataset(node, dataset) {
    node.replaceChildren();
    const isPHH = dataset.kind === 'phh';
    const phhStats = dataset.phh_stats || {}, config = dataset.config || {};
    const records = Number(dataset.records || 0), episodes = Number(dataset.episodes || 0);
    const usesEpisodes = Number(dataset.target_episodes) > 0;
    const target = isPHH ? Number(config.max_hands || dataset.max_hands || 1) : usesEpisodes ? Number(dataset.target_episodes) : Number(dataset.target_records || records || 1);
    const completed = isPHH ? Number(dataset.processed_hands ?? phhStats.processed_hands ?? 0) : usesEpisodes ? episodes : records;
    const unit = isPHH ? 'source hands processed' : usesEpisodes ? 'episodes' : 'decisions';
    cardHeading(node, dataset, `${isPHH ? 'PHH import · ' : ''}Created ${date(dataset.created)} · ${String(dataset.id).slice(0, 12)}`);
    const progressLabel = `${number(completed)} / ${number(target)} ${unit} · ${Math.min(100, completed / target * 100).toFixed(1)}%`;
    const phhProgress = dataset.status === 'ready' ? `Import finished · ${number(completed)} source hands processed` : `${progressLabel} · Maximum to process`;
    node.append(progressBar(isPHH && dataset.status === 'ready' ? target : completed, target, isPHH ? phhProgress : usesEpisodes ? progressLabel : `${progressLabel} · Original decision target`));
    const speed = Number(usesEpisodes ? dataset.episodes_per_second || 0 : dataset.records_per_second || dataset.decisions_per_second || 0);
    const facts = element('div', null, 'lab-facts');
    const plannedEpisodes = Number(dataset.planned_episodes);
    if (!isPHH && usesEpisodes && Number.isFinite(plannedEpisodes) && plannedEpisodes > 0 && plannedEpisodes !== target) {
      facts.append(element('span', `${number(plannedEpisodes)} episodes planned in full batches`));
    }
    if (isPHH) {
      facts.append(element('span', `${number(phhStats.imported_hands ?? dataset.imported_hands ?? episodes)} hands imported`));
      facts.append(element('span', `Profitable hands only: ${(config.profitable_only ?? dataset.profitable_only) ? 'On · positive net chips' : 'Off'}`));
    } else if (!usesEpisodes && dataset.episodes != null) facts.append(element('span', `${number(episodes)} episodes played`));
    facts.append(element('span', `${number(records)} training decisions saved`), element('span', `${bytes(dataset.bytes)} saved`));
    const completedBatches = dataset.completed_batches ?? dataset.completed_blocks;
    if (!isPHH && completedBatches != null) facts.append(element('span', `${number(completedBatches)} complete batches`));
    const seedGroups = dataset.seed_group_count ?? items(dataset.seed_groups).length;
    if (seedGroups) facts.append(element('span', `${number(seedGroups)} seed groups`));
    if (isPHH && dataset.finish_requested && dataset.status === 'running') facts.append(element('span', 'Finishing import at a safe checkpoint'));
    if (!isPHH && dataset.status === 'running') {
      facts.append(element('span', speed > 0 ? `${number(speed)} ${unit} / second` : 'Measuring collection speed…'));
      if (dataset.finish_requested) facts.append(element('span', 'Finishing the current balanced batch'));
      else if (completed >= target) facts.append(element('span', 'Target reached; completing the batch and seed groups'));
      else if (speed > 0) facts.append(element('span', `About ${duration((target - completed) / speed)} to target; batch completion may add time`));
    }
    node.append(facts);
    if (progressText(dataset)) node.append(element('p', progressText(dataset), 'small muted'));
    if (dataset.error) node.append(element('p', dataset.error, 'notice error'));
    const tools = element('div', null, 'run-tools');
    if (['queued', 'running'].includes(dataset.status) && !dataset.pause_requested && !dataset.finish_requested) tools.append(actionButton('datasets', dataset.id, 'pause', isPHH ? 'Pause import' : 'Pause collection'));
    if (['paused', 'interrupted', 'failed'].includes(dataset.status)) {
      if (dataset.kind !== 'import' && !isPHH) {
        const label = element('label', 'Storage limit (GB)', 'lab-resume-storage'), input = element('input');
        input.type = 'number'; input.min = '0.1'; input.max = '1000'; input.step = '0.1';
        input.value = storageOverrides.get(String(dataset.id)) ?? dataset.config?.max_disk_gb ?? 10;
        input.setAttribute('aria-label', `Storage limit for ${dataset.name || dataset.id}`);
        input.addEventListener('input', () => storageOverrides.set(String(dataset.id), input.value));
        label.append(input); tools.append(label);
      }
      tools.append(actionButton('datasets', dataset.id, 'resume', isPHH ? 'Resume import' : 'Resume collection'));
    }
    if (dataset.status !== 'ready' && !dataset.finish_requested && (records > 0 || dataset.status === 'running')) tools.append(actionButton('datasets', dataset.id, 'finish', isPHH ? 'Finish import' : 'Finish after batch'));
    if (dataset.status === 'ready') {
      tools.append(download(`/api/datasets/${encodeURIComponent(dataset.id)}/download`, 'Download dataset ZIP'));
      const use = element('button', 'Use for training', 'button primary'); use.type = 'button';
      use.addEventListener('click', () => {
        datasetIds.add(String(dataset.id));
        $('cloning-dataset-list').querySelectorAll('input').forEach(input => { input.checked = datasetIds.has(input.value); });
        renderTrainingTeachers(); updateTrainingEstimate(); location.hash = 'cloning';
      });
      tools.append(use);
    }
    if (tools.childNodes.length) node.append(tools);
    if (isPHH) {
      const sourcePath = dataset.source_path || config.source_path;
      const sourceFolder = dataset.source_folder || config.source_folder;
      if (sourcePath) node.append(element('p', `Read-only source: ${sourcePath}`, 'small muted lab-path'));
      if (sourceFolder && sourceFolder !== sourcePath) node.append(element('p', `Imported folder: ${sourceFolder}`, 'small muted lab-path'));
      node.append(element('p', `Collection: ${config.collection || dataset.collection || 'PHH'} · Teacher filter: ${config.teacher_name || dataset.teacher_name || 'All known players'}`, 'small muted'));
    }
    const detail = element('details', null, 'raw'); detail.append(element('summary', 'Teacher coverage, table sizes and dataset details'));
    if (isPHH) {
      const skipped = Object.entries(phhStats.skipped || {});
      if (skipped.length) detail.append(element('h3', 'Skipped hands'), table(skipped.map(([reason, count]) => [reason.replace(/_/g, ' '), number(count)]), ['Reason', 'Count']));
      if (phhStats.note) detail.append(element('p', phhStats.note, 'small muted'));
    }
    const rows = items(dataset.teachers).map(teacher => [teacher.name || teacher.id, String(teacher.hash || teacher.id).slice(0, 12), number(dataset.teacher_counts?.[teacher.id] || 0), ...[4, 5, 6].map(size => dataset.teacher_size_counts ? number(dataset.teacher_size_counts?.[teacher.id]?.[size] || 0) : '—')]);
    if (rows.length) detail.append(table(rows, ['Teacher', 'Version', 'Decisions', '4 players', '5 players', '6 players']));
    const totals = [4, 5, 6].map(size => `${size} players: ${number(dataset.table_size_counts?.[size] || 0)}`).join(' · ');
    detail.append(element('p', `Training decisions by table size · ${totals}`, 'small muted'));
    const coverage = element('div', null, 'form-columns');
    for (const [field, label] of [['action_counts', 'Recorded actions'], ['street_counts', 'Betting streets']]) {
      if (!dataset[field] || !Object.keys(dataset[field]).length) continue;
      const section = element('div');
      section.append(element('h3', label), table(Object.entries(dataset[field]).map(([key, count]) => [key, number(count)]), [label, 'Decisions']));
      coverage.append(section);
    }
    if (coverage.childNodes.length) detail.append(coverage);
    const inProgressNote = isPHH ? 'Pause and finish wait for a safe import checkpoint. Resume continues from the saved position. Source files remain unchanged.' : 'Pause waits for a safe checkpoint. Early finish saves after the current balanced batch. It can produce fewer than three seed groups; combine datasets or collect more before training.';
    detail.append(element('p', dataset.status === 'ready' ? 'Finished dataset. Its records are fixed for reproducible training.' : inProgressNote, 'small muted'));
    if (dataset.path || dataset.folder) detail.append(element('p', `Saved in ${dataset.path || dataset.folder}`, 'small muted lab-path'));
    node.append(detail);
  }
  function metric(value, percent = false) {
    if (!Number.isFinite(Number(value)) || value == null) return '—';
    return percent ? `${number(Number(value) * 100)}%` : number(value);
  }
  function learningCurve(history) {
    const finite = history.filter(row => Number.isFinite(Number(row.train_loss)) && row.train_loss != null);
    if (!finite.length) return null;
    const frame = element('div', null, 'lab-chart');
    const namespace = 'http://www.w3.org/2000/svg', svg = document.createElementNS(namespace, 'svg');
    svg.setAttribute('viewBox', '0 0 560 180'); svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', 'Training and validation loss by epoch. Lower loss means closer imitation.');
    const values = finite.flatMap(row => [row.train_loss, row.val_loss]).filter(value => value != null && Number.isFinite(Number(value))).map(Number);
    const maximum = Math.max(...values, 0.01) * 1.05;
    for (let step = 0; step <= 3; step++) {
      const y = 18 + step * 44;
      const line = document.createElementNS(namespace, 'line');
      for (const [key, value] of Object.entries({ x1: 46, x2: 544, y1: y, y2: y, class: 'lab-chart-grid' })) line.setAttribute(key, String(value));
      svg.append(line);
      const label = document.createElementNS(namespace, 'text'); label.setAttribute('x', '2'); label.setAttribute('y', String(y + 4)); label.textContent = (maximum * (1 - step / 3)).toFixed(2); svg.append(label);
    }
    for (const [key, color] of [['train_loss', '#55e0b0'], ['val_loss', '#8cbaff']]) {
      const points = finite.map((row, index) => {
        if (row[key] == null || !Number.isFinite(Number(row[key]))) return null;
        return [46 + index * 498 / Math.max(1, finite.length - 1), 150 - Number(row[key]) / maximum * 132];
      }).filter(Boolean);
      if (!points.length) continue;
      const line = document.createElementNS(namespace, 'polyline');
      line.setAttribute('points', points.map(point => point.join(',')).join(' ')); line.setAttribute('fill', 'none'); line.setAttribute('stroke', color); line.setAttribute('stroke-width', '2.5'); svg.append(line);
      if (points.length === 1) {
        const dot = document.createElementNS(namespace, 'circle'); dot.setAttribute('cx', points[0][0]); dot.setAttribute('cy', points[0][1]); dot.setAttribute('r', '3'); dot.setAttribute('fill', color); svg.append(dot);
      }
    }
    const axis = document.createElementNS(namespace, 'text'); axis.setAttribute('x', '46'); axis.setAttribute('y', '174'); axis.textContent = `Epoch 1 → ${finite.at(-1).epoch || finite.length}`; svg.append(axis);
    frame.append(svg, element('p', 'Green: training loss · Blue: validation loss · Lower is better', 'small muted')); return frame;
  }
  function renderTrainingJob(node, job) {
    node.replaceChildren();
    const history = items(job.history || job.progress?.history).map(row => ({ ...row, val_loss: row.validation_loss ?? row.val_loss, val_accuracy: row.validation_accuracy ?? row.val_accuracy }));
    const config = job.config || {}, metrics = job.metrics || job.progress?.metrics || {};
    const splitCounts = job.split_counts || job.progress?.split_counts;
    const raiseApproximation = job.raise_approximation || job.progress?.raise_approximation;
    const epoch = history.length ? Number(history.at(-1).epoch || history.length) : Number(job.epoch || 0);
    const datasetNames = Array.isArray(job.dataset_names) ? job.dataset_names : Object.values(job.dataset_names || {});
    cardHeading(node, job, `Created ${date(job.created)} · ${datasetNames.join(', ') || items(config.dataset_ids).join(', ') || 'Saved dataset selection'}`, 'training');
    const requestedEpochs = config.epochs || job.epochs || 0;
    const finishedEarly = job.status === 'completed' && job.stopped_early === true;
    const progressLabel = finishedEarly ? `Finished early after ${number(epoch)} of ${number(requestedEpochs)} requested epochs` : `${number(epoch)} / ${number(requestedEpochs)} epochs`;
    node.append(progressBar(finishedEarly ? 1 : epoch, finishedEarly ? 1 : requestedEpochs || 1, progressLabel));
    if (progressText(job)) node.append(element('p', progressText(job), 'small muted'));
    const facts = element('div', null, 'lab-facts');
    if (job.device || job.progress?.device || config.device) facts.append(element('span', `Device: ${job.device || job.progress?.device || config.device}`));
    if (job.records || job.progress?.records) facts.append(element('span', `${number(job.records || job.progress.records)} decisions`));
    if (metrics.test?.accuracy != null) facts.append(element('span', `Test agreement: ${metric(metrics.test.accuracy, true)}`));
    if (job.best_epoch != null) facts.append(element('span', `Best epoch: ${number(job.best_epoch)}`));
    if (history.length) {
      const latest = history.at(-1);
      facts.append(element('span', `Training agreement: ${metric(latest.train_accuracy, true)}`), element('span', `Validation agreement: ${metric(latest.val_accuracy, true)}`));
    }
    if (facts.childNodes.length) node.append(facts);
    if (job.error) node.append(element('p', job.error, 'notice error'));
    const tools = element('div', null, 'run-tools');
    if (['queued', 'running'].includes(job.status) && !job.pause_requested) tools.append(actionButton('training', job.id, 'pause', 'Pause training'));
    if (['paused', 'interrupted', 'failed'].includes(job.status)) tools.append(actionButton('training', job.id, 'resume', job.finish_requested ? 'Resume finalization' : 'Resume training'));
    if (['queued', 'running', 'paused', 'interrupted', 'failed'].includes(job.status) && job.can_finish === true && !job.finish_requested && !job.pause_requested) {
      tools.append(actionButton('training', job.id, 'finish', 'Finish now using best checkpoint'));
    }
    if (job.status === 'completed' && !job.agent_id) tools.append(actionButton('training', job.id, 'export', 'Create agent'));
    if (job.agent_id) {
      tools.append(download(`/api/agents/${encodeURIComponent(job.agent_id)}/download`, 'Download agent ZIP'));
      const launch = element('a', 'Test in arena →', 'button primary'); launch.href = '#launch'; tools.append(launch);
      node.append(element('p', 'Your student is registered. Select it and your opponents in the agent registry, then use fresh seeds to test it in the arena.', 'notice'));
    }
    if (tools.childNodes.length) node.append(tools);
    const curve = learningCurve(history); if (curve) node.append(curve);
    const details = element('details', null, 'raw'); details.append(element('summary', 'Epoch metrics, evaluation and training settings'));
    if (history.length) details.append(table(history.map(row => [row.epoch, metric(row.train_loss), metric(row.val_loss), metric(row.train_accuracy, true), metric(row.val_accuracy, true)]), ['Epoch', 'Train loss', 'Validation loss', 'Train agreement', 'Validation agreement']));
    if (splitCounts) {
      details.append(element('h3', 'Data split', 'lab-subheading'));
      details.append(table([['Training', number(splitCounts.train)], ['Validation', number(splitCounts.validation)], ['Test', number(splitCounts.test)]], ['Split', 'Decisions']));
    }
    if (metrics.validation || metrics.test) {
      details.append(element('h3', 'Held-out evaluation', 'lab-subheading'));
      const evaluations = ['validation', 'test'].filter(split => metrics[split]).map(split => [split === 'test' ? 'Test' : 'Validation', number(metrics[split].records), metric(metrics[split].loss), metric(metrics[split].accuracy, true)]);
      details.append(table(evaluations, ['Split', 'Decisions', 'Loss', 'Action agreement']));
      for (const [field, title] of [['by_teacher', 'Test agreement by teacher'], ['by_table_size', 'Test agreement by table size']]) {
        const groups = metrics.test?.[field];
        if (!groups || !Object.keys(groups).length) continue;
        const teacherNames = new Map(items(state.datasets).flatMap(dataset => items(dataset.teachers)).map(teacher => [String(teacher.id), teacher.name || teacher.id]));
        const rows = Object.entries(groups).map(([key, value]) => [field === 'by_teacher' ? teacherNames.get(key) || key : `${key} players`, number(value.records), metric(value.accuracy, true)]);
        details.append(element('h3', title, 'lab-subheading'), table(rows, [field === 'by_teacher' ? 'Teacher' : 'Table size', 'Decisions', 'Action agreement']));
      }
    }
    if (raiseApproximation) {
      details.append(element('p', `Raise-size approximation: ${number(raiseApproximation.raises)} recorded raises; mean difference ${number(raiseApproximation.mean_absolute_chips)} chips (${number(raiseApproximation.mean_absolute_bb)} big blinds), maximum ${number(raiseApproximation.maximum_absolute_chips)} chips.`, 'small muted'));
    }
    if (Object.keys(metrics).length) {
      const raw = element('details', null, 'raw'); raw.append(element('summary', 'All evaluation metrics'), element('pre', JSON.stringify(metrics, null, 2))); details.append(raw);
    }
    details.append(element('h3', 'Settings', 'lab-subheading'), element('pre', JSON.stringify(config, null, 2)));
    node.append(details);
  }
  function renderImportChoices() {
    const runs = items(state.recorded_runs), signature = JSON.stringify(runs);
    if (signature === importSignature) return;
    importSignature = signature;
    const selected = $('dataset-import-run').value;
    const options = runs.map(run => new Option(`${run.name || run.id} · ${number(run.records || run.training?.records || 0)} decisions · ${date(run.created)}`, String(run.id)));
    if (!options.length) options.push(new Option('No completed recorded runs available', ''));
    else options.unshift(new Option('Choose a recorded run', ''));
    $('dataset-import-run').replaceChildren(...options);
    if (runs.some(run => String(run.id) === selected)) $('dataset-import-run').value = selected;
    $('dataset-import-button').disabled = !runs.length;
  }
  async function refresh() {
    if (refreshing) return;
    refreshing = true;
    try {
      state = await api('/api/lab');
      window.dispatchEvent(new CustomEvent('arena-state', { detail: { source: 'lab', state } }));
      renderPHHControls(); renderEntrants(); renderTrainingChoices(); renderImportChoices();
      const datasets = items(state.datasets), jobs = items(state.training_jobs);
      $('dataset-library-count').textContent = `${datasets.filter(dataset => dataset.status === 'ready').length} finished · ${datasets.length} total`;
      $('cloning-job-count').textContent = `${jobs.length} student${jobs.length === 1 ? '' : 's'}`;
      updateCards('dataset-library', datasetCards, datasets, renderDataset, 'Your first dataset starts here', 'Choose your bots and teachers, then start a small collection to check the workflow.');
      updateCards('cloning-jobs', jobCards, jobs, renderTrainingJob, 'Train your first student', 'Choose a finished dataset, name your student and start training.');
      $('lab-error').hidden = true; $('lab-cloning-error').hidden = true;
    } catch (error) {
      $('lab-error').textContent = `Dataset and training status unavailable: ${error.message} Retrying automatically.`;
      $('lab-error').hidden = false;
      $('lab-cloning-error').textContent = $('lab-error').textContent; $('lab-cloning-error').hidden = false;
    } finally { refreshing = false; }
  }

  // Mutations are explicit button actions; polling only reads the saved state.
  async function submitForm(event, buttonId, statusId, path, payload, success) {
    event.preventDefault(); $(buttonId).disabled = true;
    try {
      const body = payload();
      message(statusId, 'Saving…');
      await api(path, body); message(statusId, success); await refresh();
    } catch (error) { message(statusId, error.message, true); }
    finally { $(buttonId).disabled = false; }
  }
  $('dataset-form').addEventListener('submit', event => submitForm(event, 'dataset-create', 'dataset-status', '/api/datasets', collectionPayload, 'Collection queued. Progress is saved in the dataset library below.'));
  $('cloning-form').addEventListener('submit', event => submitForm(event, 'cloning-create', 'cloning-status', '/api/training', trainingPayload, 'Training queued. Follow your student below.'));
  $('dataset-import-form').addEventListener('submit', event => submitForm(event, 'dataset-import-button', 'dataset-import-status', '/api/datasets/import', () => {
    const runId = $('dataset-import-run').value;
    if (!runId) throw new Error('Choose a completed recorded run.');
    return { name: $('dataset-import-name').value.trim(), run_id: runId };
  }, 'Run added to the dataset library.'));
  $('phh-preview-button').addEventListener('click', () => submitPHH(true));
  $('phh-form').addEventListener('submit', event => {
    event.preventDefault();
    submitPHH(false);
  });
  $('phh-form').addEventListener('input', event => {
    if (event.target.id === 'phh-name') return;
    if (!$('phh-preview').hidden) {
      $('phh-preview').hidden = true;
      message('phh-status', 'Settings changed. Preview again to check this selection.');
    }
  });
  $('dataset-agents-all').onclick = () => {
    for (const agent of items(state.agents)) entrants.add(String(agent.id));
    $('dataset-agent-list').querySelectorAll('input').forEach(input => { input.checked = true; }); renderCollectionTeachers(); updateCollectionEstimate();
  };
  $('dataset-agents-none').onclick = () => {
    entrants.clear(); $('dataset-agent-list').querySelectorAll('input').forEach(input => { input.checked = false; }); renderCollectionTeachers(); updateCollectionEstimate();
  };
  for (const [prefix, selection, eligible, update] of [
    ['dataset', collectionTeachers, () => eligibleCollectionTeachers, updateCollectionEstimate],
    ['cloning', trainingTeachers, () => eligibleTrainingTeachers, updateTrainingEstimate]
  ]) {
    $(prefix + '-teachers-all').onclick = () => {
      for (const id of eligible()) selection.add(id);
      $(prefix === 'dataset' ? 'dataset-teacher-list' : 'cloning-teacher-list').querySelectorAll('input[type="checkbox"]').forEach(input => { input.checked = true; }); update();
    };
    $(prefix + '-teachers-none').onclick = () => {
      selection.clear(); $(prefix === 'dataset' ? 'dataset-teacher-list' : 'cloning-teacher-list').querySelectorAll('input[type="checkbox"]').forEach(input => { input.checked = false; }); update();
    };
  }
  document.querySelectorAll('input[name="dataset-size"]').forEach(input => input.addEventListener('change', updateCollectionEstimate));
  document.querySelectorAll('input[name="cloning-size"]').forEach(input => input.addEventListener('change', updateTrainingEstimate));
  for (const id of ['dataset-target', 'dataset-deals', 'dataset-storage']) $(id).addEventListener('input', updateCollectionEstimate);
  document.addEventListener('click', async event => {
    const button = event.target.closest('button[data-lab-action]');
    if (!button || button.disabled) return;
    const { labKind: kind, labId: id, labAction: action } = button.dataset;
    const key = `${kind}/${id}`; pending.add(key); button.disabled = true;
    const statusId = kind === 'datasets' ? 'dataset-status' : 'cloning-status';
    function actionMessage(text, error = false) {
      actionMessages.set(key, { text, error }); message(statusId, text, error);
      const status = button.closest('.lab-card')?.querySelector('.lab-action-status');
      if (status) { status.textContent = text; status.hidden = false; status.classList.toggle('error-text', error); }
    }
    try {
      actionMessage(action === 'export' ? 'Packaging and checking the agent…' : 'Saving request…');
      const body = {};
      if (kind === 'datasets' && action === 'resume' && storageOverrides.has(id)) {
        const storageLimit = Number(storageOverrides.get(id));
        if (!Number.isFinite(storageLimit) || storageLimit < 0.1 || storageLimit > 1000) throw new Error('Storage limit must be between 0.1 and 1,000 GB.');
        body.max_disk_gb = storageLimit;
      }
      await api(`/api/${kind}/${encodeURIComponent(id)}/${action}`, body);
      if (kind === 'datasets' && action === 'resume') storageOverrides.delete(id);
      const phhImport = kind === 'datasets' && items(state.datasets).some(dataset => String(dataset.id) === id && dataset.kind === 'phh');
      const finishMessage = kind === 'training'
        ? 'Finish requested. The best validated checkpoint will be evaluated and saved; remaining epochs will be skipped.'
        : phhImport ? 'Finish requested. The imported records will be saved at the next safe checkpoint.'
        : 'Finish requested. The current batch will finish before saving. An early dataset with fewer than three seed groups needs other datasets before training.';
      const messages = { pause: 'Pause requested. The job will stop at a safe checkpoint.', resume: 'Job queued to resume its saved work, including finalization if requested.', finish: finishMessage, export: 'Agent created and registered. Test it in the arena with fresh seeds.' };
      actionMessage(messages[action] || 'Saved.');
    } catch (error) { actionMessage(error.message, true); }
    finally { pending.delete(key); button.disabled = false; await refresh(); }
  });
  refresh(); setInterval(refresh, 2000);
})();
