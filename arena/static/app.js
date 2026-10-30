/* Local Poker Arena. API values are rendered as text, never as HTML. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const list = value => Array.isArray(value) ? value : [];
  const selected = new Set();
  const selectedTeachers = new Set();
  let eligibleTeachers = new Set(), teacherSignature = '';
  const runNodes = new Map();
  let state = { agents: [], runs: [], leaderboard: [], profiles: [] };
  let profile = 'default', refreshing = false, refreshAgain = false;
  let rosterSignature = '', boardSignature = '';
  let previewSignature = '', estimateSignature = '';
  const defaultModeNote = $('mode-note').textContent;
  const terminal = status => /^(completed|complete|done|success|failed|error|cancelled|canceled|interrupted|paused)$/i.test(status || '');
  const display = value => value == null ? '—' : typeof value === 'object' ? JSON.stringify(value) : String(value);
  const number = value => value == null || value === '' || !Number.isFinite(Number(value)) ? '—' : Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });
  function el(tag, text, className) {
    const node = document.createElement(tag);
    if (text != null) node.textContent = display(text);
    if (className) node.className = className;
    return node;
  }
  function empty(parent, title, body) {
    const box = el('div', null, 'empty');
    box.append(el('strong', title), el('span', body));
    parent.replaceChildren(box);
  }
  function message(id, text, error = false) {
    $(id).textContent = text;
    $(id).classList.toggle('error-text', error);
  }
  async function api(path, body) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30000);
    try {
      const response = await fetch(path, { signal: controller.signal, cache: 'no-store', ...(body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }) });
      const text = await response.text();
      let data;
      try { data = text ? JSON.parse(text) : {}; } catch { throw new Error(`Server returned an unreadable response (${response.status}).`); }
      if (!response.ok) throw new Error(display(data.error || data.detail || data.message || `Request failed (${response.status}).`));
      return data;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('Request timed out. Check the queue or registry before retrying; the server may have accepted it.');
      throw error;
    } finally { clearTimeout(timeout); }
  }
  function date(value) {
    if (!value) return 'Time unavailable';
    const parsed = new Date(typeof value === 'number' && value < 1e12 ? value * 1000 : value);
    return Number.isNaN(parsed.getTime()) ? display(value) : parsed.toLocaleString();
  }
  function table(parent, rows, columns) {
    const wrap = el('div', null, 'table-wrap'), t = el('table'), head = el('tr');
    columns.forEach(c => { const th = el('th', c.label, c.numeric ? 'numeric' : ''); th.scope = 'col'; head.append(th); });
    const thead = el('thead'); thead.append(head); t.append(thead);
    const tbody = el('tbody');
    rows.forEach((row, index) => {
      const tr = el('tr');
      columns.forEach(c => {
        const value = c.get ? c.get(row, index) : row[c.key];
        const td = el('td', value instanceof Node ? null : c.numeric ? number(value) : display(value), c.numeric ? 'numeric' : '');
        if (value instanceof Node) td.append(value);
        if (c.signed && value != null && Number.isFinite(Number(value))) td.classList.add(Number(value) < 0 ? 'negative' : 'positive');
        tr.append(td);
      });
      tbody.append(tr);
    });
    t.append(tbody); wrap.append(t); parent.replaceChildren(wrap);
  }
  function downloadAgent(id) {
    const link = el('a', 'Download ZIP', 'button secondary');
    link.href = `/api/agents/${encodeURIComponent(id)}/download`;
    link.download = `agent-${id}.zip`;
    link.addEventListener('click', event => event.stopPropagation());
    return link;
  }
  function updateSelection() {
    $('selection-count').textContent = `${selected.size} agent${selected.size === 1 ? '' : 's'} selected`;
    $('library-selection-count').textContent = `${selected.size} selected for evaluation`;
    renderTeachers();
    updateStudyEstimate();
  }
  function runPayload(validateTeachers = true) {
    const agents = list(state.agents).filter(agent => selected.has(String(agent.id))).map(agent => agent.id);
    const seeds = [...new Set($('seeds').value.split(',').map(seed => seed.trim()).filter(Boolean))];
    const deals = Number($('deals-preset').value === 'custom' ? $('custom-deals').value : $('deals-preset').value);
    const mode = document.querySelector('input[name="mode"]:checked').value;
    const recordTraining = $('record-training').checked;
    const teachers = recordTraining ? agents.filter(id => selectedTeachers.has(String(id))) : [];
    if (!agents.length) throw new Error('Select at least one registered agent.');
    if (agents.length > 80) throw new Error('Select at most 80 agents per run.');
    if (!seeds.length) throw new Error('Enter at least one seed.');
    if (!Number.isSafeInteger(deals) || deals < 1) throw new Error('Hands must be a positive whole number.');
    if (validateTeachers && recordTraining && !teachers.length) throw new Error('Select at least one teacher to record training data.');
    const payload = { mode, agents, seeds, deals, profile, record_training: recordTraining, teachers };
    if (mode === 'study') {
      payload.table_sizes = Array.from(document.querySelectorAll('input[name="table-size"]:checked'), input => Number(input.value));
      payload.batches = Number($('study-batches').value);
      if (!payload.table_sizes.length) throw new Error('Choose at least one table size.');
      const minimum = Math.max(...payload.table_sizes);
      if (agents.length < minimum) throw new Error(`Select at least ${minimum} agents for the chosen table sizes.`);
      if (!Number.isSafeInteger(payload.batches) || payload.batches < 1 || payload.batches > 20) throw new Error('Batches per seed must be a whole number from 1 to 20.');
    }
    return payload;
  }
  function scheduleSignature(payload) {
    return JSON.stringify([payload.agents, payload.seeds, payload.deals, payload.table_sizes, payload.batches]);
  }
  function updateStudyEstimate() {
    const study = document.querySelector('input[name="mode"]:checked').value === 'study';
    $('study-settings').hidden = !study;
    $('study-batches').disabled = !study;
    $('mode-note').textContent = study
      ? 'Each batch balances appearances at every selected table size and spreads opponent pairings. Every lineup plays once per starting seat. Results stay separate by table size; larger tables generate more hands per bot. Start with 10 hands to check your setup.'
      : defaultModeNote;
    if (!study) return;
    try {
      const payload = runPayload(false), signature = scheduleSignature(payload);
      if (signature !== estimateSignature) {
        estimateSignature = signature;
        $('study-preview').hidden = true;
        message('study-preview-status', 'Preview the schedule to inspect planned coverage.');
      }
      let tables = 0, games = 0;
      payload.table_sizes.forEach(size => {
        const lineups = payload.agents.length === size ? 1 : payload.agents.length;
        tables += lineups;
        games += lineups * size;
      });
      const repeats = payload.seeds.length * payload.batches;
      games *= repeats; tables *= repeats;
      const overLimit = games > 25000;
      message('study-estimate', `${number(tables)} lineups · ${number(games)} games · ${number(games * payload.deals)} table hands across ${number(repeats)} batches. Each hand is counted once per table, not once per player.${overLimit ? ' Reduce seeds or batches: the limit is 25,000 games per run.' : ''}`, overLimit);
      $('study-preview-button').disabled = overLimit;
    } catch (error) {
      estimateSignature = '';
      $('study-preview').hidden = true;
      message('study-preview-status', 'Adjust the settings above before previewing.');
      message('study-estimate', error.message, true);
      $('study-preview-button').disabled = true;
    }
  }
  function showStudyPreview(preview, payload) {
    const games = preview.total_games ?? preview.games;
    const hands = preview.total_hands ?? preview.hands;
    const tables = preview.total_tables ?? preview.tables;
    previewSignature = scheduleSignature(payload);
    message('study-preview-status', `Schedule checked: ${number(tables)} lineups · ${number(games)} games · ${number(hands)} table hands.`);
    $('study-preview').hidden = false;
    const coverage = list(preview.coverage);
    table($('study-preview'), coverage, [
      { label: 'Agent', key: 'name' }, { label: 'Players', key: 'table_size', numeric: true },
      { label: 'Lineups', get: row => row.sets ?? row.tables, numeric: true },
      { label: 'Games', key: 'games', numeric: true }, { label: 'Hands / bot', key: 'hands', numeric: true }
    ]);
  }
  function updateTeacherCount() {
    $('teacher-count').textContent = `${selectedTeachers.size} teacher${selectedTeachers.size === 1 ? '' : 's'} selected`;
  }
  function renderTeachers() {
    const agents = list(state.agents).filter(agent => selected.has(String(agent.id)));
    const eligible = new Set(agents.map(agent => String(agent.id)));
    for (const id of selectedTeachers) if (!eligible.has(id)) selectedTeachers.delete(id);
    // Newly selected entrants become teachers; manual choices survive refreshes.
    for (const id of eligible) if (!eligibleTeachers.has(id)) selectedTeachers.add(id);
    eligibleTeachers = eligible;
    $('teacher-settings').hidden = !$('record-training').checked;
    const signature = JSON.stringify(agents.map(agent => [agent.id, agent.name, agent.hash]));
    if (signature !== teacherSignature) {
      teacherSignature = signature;
      $('teacher-list').replaceChildren();
      if (!agents.length) empty($('teacher-list'), 'Select agents first', 'Choose entrants in the Model library to select your teachers.');
      agents.forEach(agent => {
        const row = el('label', null, 'teacher-row'), check = el('input');
        check.type = 'checkbox'; check.value = String(agent.id); check.checked = selectedTeachers.has(check.value);
        check.addEventListener('change', () => {
          check.checked ? selectedTeachers.add(check.value) : selectedTeachers.delete(check.value);
          updateTeacherCount();
        });
        const info = el('span', null, 'agent-info');
        info.append(el('strong', agent.name || agent.id), el('small', `Version ${String(agent.hash || agent.id).slice(0, 12)}`));
        row.append(check, info); $('teacher-list').append(row);
      });
    }
    updateTeacherCount();
  }
  function renderAgents() {
    const agents = list(state.agents), ids = new Set(agents.map(a => String(a.id)));
    for (const id of selected) if (!ids.has(id)) selected.delete(id);
    const signature = JSON.stringify(agents);
    if (signature !== rosterSignature) {
      rosterSignature = signature;
      $('agent-list').replaceChildren();
      if (!agents.length) empty($('agent-list'), 'Your first contender starts here', 'Register a Python file or ZIP, then select agents for a run.');
      agents.forEach(agent => {
        const row = el('label', null, 'agent-row'), check = el('input');
        check.type = 'checkbox'; check.value = String(agent.id); check.checked = selected.has(check.value);
        check.addEventListener('change', () => { check.checked ? selected.add(check.value) : selected.delete(check.value); updateSelection(); });
        const info = el('span', null, 'agent-info');
        info.append(el('strong', agent.name || agent.id), el('small', `${String(agent.hash || 'No hash').slice(0, 12)} · ${date(agent.created)}`));
        row.append(check, el('span', String(agent.name || 'A').slice(0, 1).toUpperCase(), 'agent-avatar'), info, downloadAgent(agent.id));
        $('agent-list').append(row);
      });
    }
    $('agent-count').textContent = `${agents.length} agents`;
    updateSelection();
  }
  function renderProfiles() {
    const profiles = list(state.profiles).filter(p => p && p.id != null);
    if (!profiles.some(p => String(p.id) === profile)) profiles.unshift({ id: profile, label: profile === 'default' ? 'Default' : profile });
    const signature = JSON.stringify(profiles);
    if ($('profile').dataset.signature !== signature) {
      $('profile').replaceChildren(...profiles.map(p => new Option(p.label || p.id, String(p.id))));
      $('profile').value = profile; $('profile').dataset.signature = signature;
    }
    $('run-profile').textContent = `Profile: ${profiles.find(p => String(p.id) === profile)?.label || profile}`;
  }
  function renderBoard() {
    const rows = list(state.leaderboard), signature = JSON.stringify(rows);
    if (signature === boardSignature) return;
    boardSignature = signature;
    if (!rows.length) { empty($('leaderboard'), 'A clean slate. A fair comparison.', 'Complete a benchmark in this profile to see agent performance here.'); return; }
    table($('leaderboard'), rows, [
      { label: '#', get: (_, i) => String(i + 1).padStart(2, '0') }, { label: 'Agent', get: r => r.name || r.id },
      { label: 'Chips', key: 'chips', numeric: true, signed: true }, { label: 'Hands', key: 'hands', numeric: true },
      { label: 'mbb / hand', key: 'mbb', numeric: true, signed: true },
      { label: 'Sets', key: 'sets', numeric: true }, { label: 'Set win rate', get: r => r.win_rate == null ? '—' : typeof r.win_rate === 'number' ? `${number(r.win_rate)}%` : r.win_rate },
      { label: 'Failures', key: 'failures', numeric: true },
      { label: 'Export', get: r => downloadAgent(r.id) }
    ]);
  }
  function renderTournament() {
    const runs = list(state.runs).filter(r => r.mode === 'tournament');
    const select = $('tournament-run'), chosen = select.value || 'latest';
    select.replaceChildren(new Option('Latest completed tournament', 'latest'), ...runs.map(r => new Option(`${date(r.created)} · ${r.status} · ${r.id.slice(0, 8)}`, r.id)));
    select.value = chosen === 'latest' || runs.some(r => r.id === chosen) ? chosen : 'latest';
    const run = select.value === 'latest' ? runs.filter(r => r.status === 'completed').sort((a, b) => String(b.finished).localeCompare(String(a.finished)))[0] : runs.find(r => r.id === select.value);
    if (!run) {
      $('tournament-note').textContent = 'No completed tournament yet. Queued and running tournaments can be selected above.';
      empty($('tournament-board'), 'No tournament standings yet', 'Queue a tournament between selected agents.'); return;
    }
    $('tournament-note').textContent = `${run.id} · ${run.status} · ${run.config.deals} hands/game · ${run.config.seeds.length} seed(s). ${run.status === 'completed' ? 'Final local standings.' : 'Provisional results; failed/cancelled runs are not final rankings.'}`;
    const rows = list(run.leaderboard).slice().sort((a, b) => b.placement_points - a.placement_points || b.playoff_points - a.playoff_points);
    if (!rows.length) { empty($('tournament-board'), 'Waiting for round results', display(run.progress)); return; }
    table($('tournament-board'), rows, [
      {label:'Rank',get:r => r.rank ?? 1 + rows.filter(o => o.placement_points > r.placement_points || (o.placement_points === r.placement_points && o.playoff_points > r.playoff_points)).length,numeric:true},
      {label:'Agent',key:'name'}, {label:'Round points',key:'placement_points',numeric:true},
      {label:'Game points',key:'game_points',numeric:true}, {label:'Playoff points',key:'playoff_points',numeric:true},
      {label:'Chips',key:'chips',numeric:true,signed:true}, {label:'Failures',key:'failures',numeric:true},
      {label:'Export',get:r => downloadAgent(r.id)}
    ]);
  }
  function studyMatrix(parent, run) {
    const rows = list(run.study?.rows), agents = new Map();
    rows.forEach(row => {
      if (!agents.has(row.id)) agents.set(row.id, { id: row.id, name: row.name, sizes: {} });
      agents.get(row.id).sizes[row.table_size] = row;
    });
    if (!rows.length) { empty(parent, 'Waiting for study results', 'Each bot will have a separate result for every selected table size.'); return; }
    const columns = [{ label: 'Agent', key: 'name' }];
    [4, 5, 6].forEach(size => columns.push({ label: `${size} players · bb/100`, get: agent => {
      const row = agent.sizes[size], cell = el('div', null, 'study-rate');
      if (!row || !row.hands) {
        cell.append(el('strong', '—'), el('small', row ? 'No completed games' : 'Not included'));
        return cell;
      }
      const value = row.bb_per_100;
      cell.append(el('strong', number(value), value == null ? '' : value < 0 ? 'negative' : 'positive'));
      cell.append(el('small', `${number(row.hands)} hands · ${number(row.failures || 0)} failures`));
      return cell;
    }}));
    table(parent, Array.from(agents.values()), columns);
  }
  function studyCoverage(parent, run) {
    const rows = list(run.study?.rows);
    if (!rows.length) { empty(parent, 'No coverage yet', 'The study records games, hands and teacher decisions separately for each size.'); return; }
    table(parent, rows, [
      { label: 'Agent', key: 'name' }, { label: 'Players', key: 'table_size', numeric: true },
      { label: 'bb / 100', key: 'bb_per_100', numeric: true, signed: true },
      { label: 'Hands', key: 'hands', numeric: true }, { label: 'Games', key: 'games', numeric: true },
      { label: 'Complete sets', key: 'completed_sets', numeric: true },
      { label: 'Failures', key: 'failures', numeric: true }, { label: 'Decisions', key: 'decisions', numeric: true },
      { label: 'Complete batches', key: 'completed_batches', numeric: true },
      { label: 'Observed batch range · bb/100', get: row => row.completed_batches >= 2 && row.batch_min_bb_per_100 != null && row.batch_max_bb_per_100 != null
        ? `${number(row.batch_min_bb_per_100)} to ${number(row.batch_max_bb_per_100)}` : 'Needs 2 complete batches' }
    ]);
  }
  function renderStudy() {
    const runs = list(state.runs).filter(run => run.mode === 'study').sort((left, right) => String(right.created).localeCompare(String(left.created)));
    const select = $('study-run'), chosen = select.value || 'latest';
    const optionsSignature = JSON.stringify(runs.map(run => [run.id, run.created, run.status]));
    if (select.dataset.signature !== optionsSignature) {
      select.replaceChildren(new Option('Latest study', 'latest'), ...runs.map(run => new Option(`${date(run.created)} · ${run.status} · ${run.id.slice(0, 8)}`, run.id)));
      select.dataset.signature = optionsSignature;
    }
    select.value = chosen === 'latest' || runs.some(run => run.id === chosen) ? chosen : 'latest';
    const run = select.value === 'latest' ? runs[0] : runs.find(candidate => candidate.id === select.value);
    if (!run) {
      $('study-note').textContent = 'Create a Table-size study to compare your bots at 4, 5 and 6 players.';
      empty($('study-board'), 'Every table size, a separate result', 'Select your bots and choose Table-size study in Configure a run.');
      $('study-coverage').replaceChildren();
      return;
    }
    const study = run.study || {};
    const resultSignature = JSON.stringify([run.id, run.status, run.pause_requested, study]);
    if ($('study-board').dataset.signature === resultSignature) return;
    $('study-board').dataset.signature = resultSignature;
    const progress = `${number(study.completed_games || 0)} / ${number(study.total_games)} games · ${number(study.completed_hands || 0)} / ${number(study.total_hands)} table hands`;
    const coverageNote = run.status === 'completed' ? 'Complete coverage for this schedule.' : 'Partial coverage: results can change as the study continues.';
    const pauseNote = run.pause_requested ? ' Pausing after the current game.' : '';
    $('study-note').textContent = `${run.id} · ${run.status} · ${progress}. ${coverageNote}${pauseNote}`;
    studyMatrix($('study-board'), run);
    studyCoverage($('study-coverage'), run);
  }
  function raw(parent, title, value, open = false) {
    const details = el('details', null, 'raw'); details.open = open;
    details.append(el('summary', title), el('pre', value == null ? 'Not provided.' : JSON.stringify(value, null, 2))); parent.append(details);
  }
  function section(parent, title) {
    const node = el('section', null, 'detail-section'); node.append(el('h3', title)); parent.append(node); return node;
  }
  function gameRows(game, agents) {
    const chips = game.chips || [], verdicts = game.verdicts || [];
    const count = Math.max(agents.length, Object.keys(chips).length, Object.keys(verdicts).length);
    return Array.from({ length: count }, (_, i) => ({ name: agents[i]?.name || agents[i]?.id || `Seat ${i + 1}`, chips: chips[i] ?? chips[agents[i]?.id], verdict: verdicts[i] ?? verdicts[agents[i]?.id] }));
  }
  function historyHands(data) {
    if (Array.isArray(data)) return data;
    for (const key of ['hands', 'hand_histories', 'histories', 'history', 'deals']) {
      if (Array.isArray(data?.[key])) return data[key];
      if (data?.[key] && typeof data[key] === 'object') return Object.values(data[key]);
    }
    return data && typeof data === 'object' ? [data] : [];
  }
  function renderHand(parent, hand, history, game) {
    parent.replaceChildren();
    const cards = hand?.board ?? [...(hand?.events || [])].reverse().find(e => e.board)?.board ?? hand?.community_cards ?? hand?.cards ?? hand?.public_cards;
    const board = section(parent, 'Cards');
    if (Array.isArray(cards) && cards.every(c => typeof c === 'string' || typeof c === 'number')) {
      const row = el('div', null, 'history-board');
      cards.forEach(c => row.append(el('span', c, /[♥♦hd]$/i.test(String(c)) ? 'playing-card red' : 'playing-card')));
      board.append(row);
    } else raw(board, 'Card data', cards, true);
    const grid = el('div', null, 'history-grid'); parent.append(grid);
    const seats = section(grid, 'Seat map & private cards');
    const mapping = hand?.seat_map || [], agents = game.agents || [];
    const names = player => agents[player]?.name || `Player ${player}`;
    const seatTarget = el('div'); seats.append(seatTarget);
    table(seatTarget, mapping.map((player, seat) => ({ seat, name:names(player), cards:(hand?.holes?.[seat] || []).join(' '), delta:hand?.deltas_by_bot?.[player] })), [
      {label:'Seat',key:'seat',numeric:true},{label:'Agent',key:'name'},{label:'Cards',key:'cards'},{label:'Δ chips',key:'delta',numeric:true,signed:true}
    ]);
    if ((hand?.holes ?? hand?.hole_cards) != null) raw(seats, 'Hole cards (spectator only)', hand.holes ?? hand.hole_cards);
    const deltas = section(grid, 'Chip deltas');
    raw(deltas, 'Per-player changes', hand?.deltas_by_bot ?? hand?.deltas ?? hand?.chip_deltas ?? hand?.payoffs ?? hand?.rewards ?? hand?.chips, true);
    const eventsBox = section(parent, 'Hand events');
    const events = hand?.events ?? hand?.actions ?? hand?.history ?? hand?.steps;
    if (Array.isArray(events) && events.length) {
      const ol = el('ol', null, 'events'); events.forEach(event => {
        let text = display(event);
        if(event.type === 'action') text = `${event.street} · ${names(mapping[event.seat])} (seat ${event.seat}): ${event.action}, amount ${event.amount} · pot ${event.pot}`;
        if(event.type === 'street') text = `${event.street.toUpperCase()} · ${(event.board || []).join(' ')}`;
        if(event.type === 'blinds') text = `Blinds · seat ${event.sb_seat}: ${event.sb} · seat ${event.bb_seat}: ${event.bb}`;
        if(event.type === 'hand_end') text = `Hand complete · board ${(event.board || []).join(' ')} · ${list(event.pots).map(p => `pot ${p.amount} → ${list(p.winners).map(s => names(mapping[s])).join(', ')}`).join(' / ')}`;
        ol.append(el('li', text));
      }); eventsBox.append(ol);
    } else eventsBox.append(el('p', 'No structured events supplied. Expand the raw hand below to inspect all fields.', 'muted small'));
    raw(parent, 'Raw hand JSON · all fields', hand);
  }
  function historyViewer(parent, runId, games) {
    const box = section(parent, 'Hand history explorer');
    const available = games.filter(g => g.game.history_file);
    if (!available.length) { box.append(el('p', 'Hand histories will appear when games provide history files.', 'muted small')); return; }
    const controls = el('div', null, 'history-controls'), gameLabel = el('label', 'Game'), gameSelect = el('select');
    available.forEach((g, i) => gameSelect.add(new Option(g.label, String(i))));
    gameLabel.append(gameSelect);
    const handLabel = el('label', 'Hand'), handSelect = el('select'); handLabel.append(handSelect);
    const download = el('a', '↓ Raw JSON'); download.download = 'history.json';
    controls.append(gameLabel, handLabel, download);
    const content = el('div'); box.append(controls, content);
    let request = 0, hands = [], history = null;
    const show = () => renderHand(content, hands[Number(handSelect.value)], history, available[Number(gameSelect.value)]);
    handSelect.addEventListener('change', show);
    async function load() {
      const version = ++request, entry = available[Number(gameSelect.value)];
      const path = `/api/history/${encodeURIComponent(runId)}/${encodeURIComponent(entry.game.history_file)}`;
      download.href = path; handSelect.replaceChildren(); handSelect.disabled = true;
      content.replaceChildren(el('p', 'Loading hand history…', 'muted'));
      try {
        const data = await api(path); if (version !== request || !box.isConnected) return;
        history = data; hands = historyHands(data);
        hands.forEach((h, i) => handSelect.add(new Option(`Hand ${h?.hand_id ?? h?.hand_number ?? i + 1}`, String(i))));
        handSelect.disabled = !hands.length;
        if (hands.length) show(); else empty(content, 'No hands in this file', 'Download the raw JSON to inspect its contents.');
      } catch (error) {
        if (version !== request) return;
        const retry = el('button', 'Retry history', 'button secondary'); retry.type = 'button'; retry.onclick = load;
        content.replaceChildren(el('p', error.message, 'error-text'), retry);
      }
    }
    gameSelect.addEventListener('change', load); load();
  }
  function renderDetail(body, data, runId) {
    body.replaceChildren();
    if (data.error) body.append(el('div', display(data.error), 'notice error'));
    raw(body, 'Run configuration', data.config);
    const standings = list(data.leaderboard);
    if (data.mode === 'study') {
      const box = section(body, 'Results by table size'), target = el('div'), coverage = el('div');
      if (data.status !== 'completed') box.append(el('p', 'Partial coverage. Completed games are preserved when you pause or resume this study.', 'notice'));
      box.append(target, coverage); studyMatrix(target, data); studyCoverage(coverage, data);
    } else if (standings.length) {
      const box = section(body, 'Run standings'), target = el('div'); box.append(target);
      table(target, standings, [{label:'Rank',key:'rank',numeric:true}, { label: 'Agent', get: r => r.name || r.id }, ...['placement_points', 'game_points', 'playoff_points', 'chips', 'failures'].map(key => ({ key, label: key.replaceAll('_', ' '), numeric: true }))]);
    }
    const results = list(data.results), allGames = [];
    if (!results.length) empty(body.appendChild(el('div')), 'No game results yet', 'Queued and running experiments update automatically.');
    results.forEach((result, ri) => {
      const title = data.mode === 'study'
        ? `${number(result.table_size ?? list(result.agents).length)} players · Lineup ${result.table ?? ri + 1} · Seed ${result.seed ?? '—'}`
        : `Round ${result.round ?? ri + 1} · Table ${result.table ?? '—'} · Seed ${result.seed ?? '—'}`;
      const box = section(body, title);
      list(result.games).forEach((game, gi) => {
        const entry = { game, agents: list(result.agents), label: `${title} / Game ${gi + 1}` };
        allGames.push(entry);
        const detail = el('details', null, 'game');
        detail.append(el('summary', `Game ${gi + 1} · ${number(game.num_hands)} hands`));
        const target = el('div'); detail.append(target);
        table(target, gameRows(game, entry.agents), [{ label: 'Agent / seat', key: 'name' }, { label: 'Chips', key: 'chips', numeric: true, signed: true }, { label: 'Verdict', key: 'verdict' }]);
        raw(detail, 'Execution logs', game.logs);
        box.append(detail);
      });
      if (data.mode !== 'study' && list(result.standings).length) raw(box, 'Table standings', result.standings);
    });
    historyViewer(body, runId, allGames);
  }
  function renderTraining(parent, run) {
    parent.hidden = !run.config?.record_training;
    if (parent.hidden) return;
    const training = run.training || {};
    const signature = JSON.stringify([training, run.config.teachers, run.status]);
    if (parent.dataset.signature === signature) return;
    parent.dataset.signature = signature;
    parent.replaceChildren(el('h3', 'Training data'));
    parent.append(el('p', `Recording enabled · ${number(training.records || 0)} saved decisions`, 'training-total'));
    const teachers = list(training.teachers).length ? training.teachers : list(run.config.teachers).map(id => ({ id, name: list(state.agents).find(agent => String(agent.id) === String(id))?.name || id }));
    const target = el('div'); parent.append(target);
    const columns = [
      { label: 'Teacher', get: teacher => teacher.name || teacher.id },
      { label: 'Saved decisions', get: teacher => training.teacher_counts?.[teacher.id] || 0, numeric: true }
    ];
    if (run.mode === 'study') [4, 5, 6].forEach(size => columns.push({ label: `${size} players`, get: teacher => training.teacher_size_counts?.[teacher.id]?.[size] || 0, numeric: true }));
    table(target, teachers, columns);
    if (training.excluded_records) parent.append(el('p', `${number(training.excluded_records)} decisions excluded from the training dataset.`, 'small muted'));
    if (run.status === 'completed' && training.records > 0 && training.download_url) {
      const download = el('a', 'Download training data', 'button secondary');
      download.href = `/api/runs/${encodeURIComponent(run.id)}/training/download`;
      download.download = `training-${run.id}.zip`;
      const downloads = el('div', null, 'training-downloads'); downloads.append(download);
      [4, 5, 6].forEach(size => {
        if (!training.download_urls_by_size?.[size] || !training.table_size_counts?.[size]) return;
        const sizeDownload = el('a', `${size} players · ${number(training.table_size_counts[size])} decisions`, 'button secondary');
        sizeDownload.href = `/api/runs/${encodeURIComponent(run.id)}/training/download?table_size=${size}`;
        sizeDownload.download = `training-${run.id}-${size}-players.zip`;
        downloads.append(sizeDownload);
      });
      parent.append(downloads, el('p', 'ZIP includes decision records (JSONL) and a dataset manifest. A student model can be trained from this data separately.', 'small muted'));
    } else {
      let note = 'Data will be available to download after this run completes.';
      if (run.status === 'completed') note = 'No usable teacher decisions were saved in this run.';
      else if (run.mode === 'study' && terminal(run.status)) note = 'Recording is incomplete. Resume this study to finish its schedule and enable training downloads.';
      else if (terminal(run.status)) note = 'This run did not complete. Its recording is incomplete and is not available as a training download.';
      parent.append(el('p', note, 'small muted'));
    }
  }
  async function loadDetail(entry, force = false) {
    if (entry.loading || (!force && entry.loaded && terminal(entry.run.status) && entry.loadedStatus === entry.run.status)) return;
    entry.loading = true;
    if (!entry.loaded) entry.detail.replaceChildren(el('p', 'Loading run details…', 'muted'));
    try {
      const data = await api(`/api/runs/${encodeURIComponent(entry.run.id)}`);
      if (!entry.root.isConnected) return;
      renderTraining(entry.training, data);
      const signature = JSON.stringify(data);
      if (signature !== entry.signature) {
        // Progress-only updates must not reset an open game or selected hand.
        const contentSignature = JSON.stringify([data.results, data.leaderboard, data.study, data.status, data.error, data.config]);
        if (contentSignature !== entry.contentSignature) { renderDetail(entry.detail, data, entry.run.id); entry.contentSignature = contentSignature; }
        entry.signature = signature;
      }
      entry.loaded = true; entry.loadedStatus = data.status; entry.detailError.textContent = '';
    } catch (error) { entry.detailError.textContent = `Could not refresh details: ${error.message}`; }
    finally { entry.loading = false; }
  }
  function renderRuns() {
    const runs = list(state.runs), ids = new Set(runs.map(r => String(r.id)));
    for (const [id, entry] of runNodes) if (!ids.has(id)) { entry.root.remove(); runNodes.delete(id); }
    if (!runs.length) { empty($('run-list'), 'The table is waiting', 'Select your agents and queue a benchmark, tournament or table-size study. Runs remain in the server queue when you leave this page.'); return; }
    $('run-list').querySelector(':scope > .empty')?.remove();
    runs.forEach((run, runIndex) => {
      const id = String(run.id); let entry = runNodes.get(id);
      if (!entry) {
        const root = el('details', null, 'run'), summary = el('summary'), title = el('span', null, 'run-title');
        const name = el('strong'), meta = el('small'), badge = el('span', null, 'tag'), progress = el('span', null, 'run-progress');
        title.append(name, meta); summary.append(title, progress, badge);
        const body = el('div', null, 'run-body'), tools = el('div', null, 'run-tools'), cancel = el('button', 'Cancel run', 'button danger'), retry = el('button', 'Refresh details', 'button secondary');
        const pause = el('button', 'Pause after current game', 'button secondary'), resume = el('button', 'Resume study', 'button primary');
        cancel.type = retry.type = pause.type = resume.type = 'button'; tools.append(retry, pause, resume, cancel);
        const detailError = el('div', null, 'error-text small'); detailError.setAttribute('role', 'status');
        const training = el('section', null, 'detail-section training-result'), detail = el('div');
        training.hidden = true; body.append(tools, detailError, training, detail); root.append(summary, body);
        entry = { root, name, meta, badge, progress, cancel, pause, resume, training, detail, detailError, run, loaded: false, loading: false };
        const current = entry;
        root.addEventListener('toggle', () => { if (root.open) loadDetail(current, true); });
        retry.onclick = () => loadDetail(current, true);
        cancel.onclick = async () => {
          cancel.disabled = true;
          try { await api(`/api/runs/${encodeURIComponent(id)}/cancel`, {}); detailError.textContent = 'Cancellation requested.'; await refresh(); }
          catch (error) { detailError.textContent = error.message; }
          finally { cancel.disabled = false; }
        };
        pause.onclick = async () => {
          pause.disabled = true;
          try { await api(`/api/runs/${encodeURIComponent(id)}/pause`, {}); detailError.textContent = 'Pause requested. Completed games will be preserved.'; await refresh(); }
          catch (error) { detailError.textContent = error.message; }
          finally { pause.disabled = false; }
        };
        resume.onclick = async () => {
          resume.disabled = true;
          try { await api(`/api/runs/${encodeURIComponent(id)}/resume`, {}); detailError.textContent = 'Study queued to resume from its saved progress.'; await refresh(); }
          catch (error) { detailError.textContent = error.message; }
          finally { resume.disabled = false; }
        };
        runNodes.set(id, entry);
      }
      const currentPosition = $('run-list').children[runIndex];
      if (currentPosition !== entry.root) $('run-list').insertBefore(entry.root, currentPosition || null);
      const previousStatus = entry.run.status; entry.run = run;
      const modeName = { tournament: 'Tournament', benchmark: 'Benchmark', study: 'Table-size study' }[run.mode] || run.mode;
      entry.name.textContent = `${modeName} · ${id}`;
      entry.meta.textContent = `${date(run.created)} · ${list(run.config?.agents).length} agents · ${number(run.config?.deals)} hands`;
      if (run.config?.record_training) entry.meta.textContent += ` · Recording enabled · ${number(run.training?.records || 0)} saved decisions`;
      const status = String(run.status || 'unknown');
      entry.badge.textContent = status; entry.badge.className = `tag status-${status.toLowerCase().replace(/[^a-z]/g, '')}`;
      entry.progress.textContent = run.progress == null ? '' : display(run.progress);
      if (run.mode === 'study') {
        entry.meta.textContent += ` · ${list(run.config?.table_sizes).join('/')} players · ${number(run.study?.completed_games || 0)}/${number(run.study?.total_games)} games`;
        if (run.pause_requested) entry.progress.textContent = 'Pausing after current game';
      }
      entry.cancel.hidden = terminal(status);
      entry.pause.hidden = run.mode !== 'study' || !['queued', 'running'].includes(status) || !!run.pause_requested;
      entry.resume.hidden = run.mode !== 'study' || !['paused', 'interrupted', 'failed', 'cancelled'].includes(status);
      renderTraining(entry.training, run);
      if (entry.root.open) loadDetail(entry, previousStatus !== run.status);
    });
  }
  async function refresh() {
    if (refreshing) { refreshAgain = true; return; }
    refreshing = true; const requestedProfile = profile;
    try {
      const next = await api(`/api/state?profile=${encodeURIComponent(requestedProfile)}`);
      if (requestedProfile !== profile) { refreshAgain = true; return; }
      state = next || {}; profile = state.profile || 'default'; renderProfiles(); renderAgents(); renderBoard(); renderTournament(); renderStudy(); renderRuns();
      window.dispatchEvent(new CustomEvent('arena-state', { detail: { source: 'arena', state } }));
      $('stat-agents').textContent = number(list(state.agents).length);
      $('stat-active').textContent = number(list(state.runs).filter(r => !terminal(r.status)).length + Number(state.lab_active || 0));
      $('stat-complete').textContent = number(list(state.runs).filter(r => /^(completed|complete|done|success)$/i.test(r.status)).length);
      const hands = list(state.leaderboard).filter(r => r.hands != null && Number.isFinite(Number(r.hands)));
      $('stat-hands').textContent = hands.length ? number(hands.reduce((sum, r) => sum + Number(r.hands), 0)) : '—';
      $('connection').textContent = 'Live · local server'; $('connection').classList.add('online');
      $('state-error').hidden = true;
    } catch (error) {
      $('connection').textContent = 'Connection interrupted'; $('connection').classList.remove('online');
      $('state-error').textContent = `Unable to refresh: ${error.message} Showing the last available data. Retrying automatically.`;
      $('state-error').hidden = false;
    } finally {
      refreshing = false;
      if (refreshAgain) { refreshAgain = false; refresh(); }
    }
  }
  $('profile').addEventListener('change', () => {
    profile = $('profile').value; boardSignature = '';
    $('leaderboard').replaceChildren(el('p', 'Loading selected profile…', 'muted'));
    $('stat-hands').textContent = '—'; $('run-profile').textContent = `Profile: ${profile}`; refresh();
  });
  $('select-all').onclick = () => { list(state.agents).forEach(a => selected.add(String(a.id))); $('agent-list').querySelectorAll('input').forEach(c => { c.checked = true; }); updateSelection(); };
  $('select-none').onclick = () => { selected.clear(); $('agent-list').querySelectorAll('input').forEach(c => { c.checked = false; }); updateSelection(); };
  $('record-training').onchange = renderTeachers;
  $('teachers-all').onclick = () => {
    eligibleTeachers.forEach(id => selectedTeachers.add(id));
    $('teacher-list').querySelectorAll('input').forEach(check => { check.checked = true; });
    updateTeacherCount();
  };
  $('teachers-none').onclick = () => {
    selectedTeachers.clear();
    $('teacher-list').querySelectorAll('input').forEach(check => { check.checked = false; });
    updateTeacherCount();
  };
  $('deals-preset').onchange = () => { const custom = $('deals-preset').value === 'custom'; $('custom-deals-label').hidden = !custom; $('custom-deals').required = custom; updateStudyEstimate(); };
  document.querySelectorAll('input[name="mode"], input[name="table-size"]').forEach(input => input.addEventListener('change', updateStudyEstimate));
  ['seeds', 'custom-deals', 'study-batches'].forEach(id => $(id).addEventListener('input', updateStudyEstimate));
  $('study-preview-button').onclick = async () => {
    let payload;
    try { payload = runPayload(false); }
    catch (error) { message('study-preview-status', error.message, true); return; }
    $('study-preview-button').disabled = true;
    message('study-preview-status', 'Checking schedule and coverage…');
    try {
      const preview = await api('/api/study/preview', payload);
      if (scheduleSignature(runPayload(false)) !== scheduleSignature(payload)) {
        message('study-preview-status', 'Settings changed while the preview loaded. Preview again for the current selection.');
        return;
      }
      showStudyPreview(preview, payload);
    } catch (error) { message('study-preview-status', error.message, true); }
    finally { updateStudyEstimate(); }
  };
  $('agent-files').onchange = () => { $('file-preview').textContent = Array.from($('agent-files').files).map(f => f.name).join(' · ') || 'Each file is registered as a separate agent.'; };
  function base64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result).split(',')[1]);
      reader.onerror = () => reject(new Error(`Could not read ${file.name}.`));
      reader.onabort = () => reject(new Error(`Reading ${file.name} was cancelled.`));
      reader.readAsDataURL(file);
    });
  }
  $('upload-form').addEventListener('submit', async event => {
    event.preventDefault(); const files = Array.from($('agent-files').files), prefix = $('agent-name').value.trim();
    if (!files.length) return message('upload-status', 'Choose at least one .py or .zip file.', true);
    if (files.some(f => !/\.(py|zip)$/i.test(f.name))) return message('upload-status', 'Only .py and .zip files can be registered.', true);
    $('upload-button').disabled = true; $('agent-files').disabled = true;
    let success = 0; const failures = [];
    for (let i = 0; i < files.length; i++) {
      const file = files[i], stem = file.name.replace(/\.(py|zip)$/i, '');
      message('upload-status', `Registering ${i + 1}/${files.length}: ${file.name}`);
      try {
        await api('/api/agents', { name: prefix ? files.length === 1 ? prefix : `${prefix} · ${stem}` : stem, filename: file.name, content: await base64(file) }); success++;
      } catch (error) { failures.push(`${file.name}: ${error.message}`); }
    }
    message('upload-status', `${success}/${files.length} agents registered.${failures.length ? ` Failed files: ${failures.join(' | ')} Successful uploads are already saved; retry only failed files.` : ''}`, !!failures.length);
    $('agent-files').value = ''; $('file-preview').textContent = 'Each file is registered as a separate agent.';
    $('upload-button').disabled = false; $('agent-files').disabled = false; refresh();
  });
  $('run-form').addEventListener('submit', async event => {
    event.preventDefault();
    let payload;
    try { payload = runPayload(); }
    catch (error) { message('run-status', error.message, true); return; }
    $('queue-button').disabled = true;
    try {
      if (payload.mode === 'study' && previewSignature !== scheduleSignature(payload)) {
        message('run-status', 'Checking the study schedule…');
        const preview = await api('/api/study/preview', payload);
        if (scheduleSignature(runPayload()) !== scheduleSignature(payload)) throw new Error('Run settings changed. Review the updated estimate and queue again.');
        showStudyPreview(preview, payload);
      }
      await api('/api/runs', payload);
      message('run-status', 'Run queued. Follow its progress in Experiments → Evaluation queue & history.'); refresh();
    } catch (error) { message('run-status', error.message, true); }
    finally { $('queue-button').disabled = false; }
  });
  $('tournament-run').onchange = renderTournament;
  $('study-run').onchange = renderStudy;
  empty($('leaderboard'), 'Connecting to your arena', 'Loading benchmark results…');
  updateStudyEstimate();
  refresh(); setInterval(refresh, 2000);
})();
