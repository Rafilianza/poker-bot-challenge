/* GRPO training workspace. API values are rendered as text, never as HTML. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const list = value => Array.isArray(value) ? value : [];
  const finite = value => value != null && value !== '' && Number.isFinite(Number(value));
  const number = value => finite(value) ? Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 }) : '—';
  const precise = value => finite(value) ? Number(value).toLocaleString(undefined, { maximumFractionDigits: 6 }) : '—';
  const opponents = new Set(), pending = new Set(), jobCards = new Map(), actionMessages = new Map();
  let state = { checkpoints: [], agents: [], grpo_jobs: [] };
  let refreshing = false, checkpointSignature = '', opponentSignature = '';

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
    box.append(element('strong', title), element('span', text)); parent.replaceChildren(box);
  }
  async function api(path, body) {
    const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 60000);
    try {
      const options = { signal: controller.signal, cache: 'no-store' };
      if (body !== undefined) {
        options.method = 'POST'; options.headers = { 'Content-Type': 'application/json' };
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
      if (error.name === 'AbortError') throw new Error('The request timed out. Check the GRPO job list before retrying; the server may have accepted it.');
      throw error;
    } finally { clearTimeout(timeout); }
  }
  function date(value) {
    if (!value) return '—';
    const parsed = new Date(typeof value === 'number' && value < 1e12 ? value * 1000 : value);
    return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString();
  }
  function duration(value) {
    const seconds = Math.max(0, Math.floor(Number(value || 0)));
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor(seconds % 3600 / 60)}m`;
  }
  function inputNumber(id, title, minimum, maximum, whole = false) {
    const value = Number($(id).value);
    if ($(id).value.trim() === '' || !Number.isFinite(value) || value < minimum || value > maximum || (whole && !Number.isSafeInteger(value))) {
      throw new Error(`${title} must be ${whole ? 'a whole number' : 'a number'} from ${precise(minimum)} to ${precise(maximum)}.`);
    }
    return value;
  }
  function tableSizes() {
    return Array.from(document.querySelectorAll('input[name="grpo-size"]:checked'), input => Number(input.value));
  }
  function sourceCheckpoint() {
    return list(state.checkpoints).find(checkpoint => String(checkpoint.id) === $('grpo-source').value);
  }
  function rewardFields() {
    const groupSize = Number($('grpo-group-size').value);
    const groupEpisodes = groupSize * 100;
    for (const id of ['grpo-target', 'grpo-rollout', 'grpo-milestone-interval']) {
      $(id).min = String(groupEpisodes);
      $(id).step = String(groupEpisodes);
    }
    const weight = Number($('grpo-placement-weight').value);
    message('grpo-reward-note', `${number(weight)}% game placement + ${number(100 - weight)}% bounded chip profit. Each group plays ${number(groupSize)} matched 100-hand games (${number(groupEpisodes)} episodes). Above-average attempts receive positive relative advantage; below-average attempts receive negative relative advantage.`);
  }
  function milestoneSettings() {
    const enabled = $('grpo-milestones-enabled').checked;
    return {
      enabled,
      interval: enabled ? inputNumber('grpo-milestone-interval', 'Milestone interval', 1, 10000000, true) : 50000,
      eval_hands: 100,
      eval_seeds: enabled ? inputNumber('grpo-milestone-seeds', 'Milestone seed groups', 1, 20, true) : 3
    };
  }
  function payload() {
    const checkpoint = sourceCheckpoint(), sizes = tableSizes();
    if (!checkpoint || !checkpoint.eligible || checkpoint.status !== 'completed') throw new Error('Choose an eligible, completed Behaviour Cloning checkpoint.');
    if (!sizes.length) throw new Error('Choose at least one table size.');
    const milestones = milestoneSettings();
    const minimum = milestones.enabled ? 5 : Math.max(...sizes) - 1;
    if (opponents.size < minimum || opponents.size > 60) throw new Error(milestones.enabled
      ? 'Choose 5–60 distinct opponents. Milestone evaluations always include 6-player tables, even when training uses smaller tables.'
      : `Choose ${minimum}–60 distinct opponents for these table sizes. The learner fills one seat.`);
    if (state.development_mode && $('grpo-device').value === 'mps') throw new Error('This test workspace uses CPU to leave the main training GPU available.');
    const settings = {
      name: $('grpo-name').value.trim(), source_training_id: String(checkpoint.id), opponents: [...opponents], table_sizes: sizes,
      group_size: inputNumber('grpo-group-size', 'Attempts per group', 2, 8, true),
      target_episodes: inputNumber('grpo-target', 'Training episodes', 1, 10000000, true),
      rollout_episodes: inputNumber('grpo-rollout', 'Episodes per rollout', 1, 10000, true),
      update_epochs: inputNumber('grpo-update-epochs', 'Update passes', 1, 20, true),
      batch_size: inputNumber('grpo-batch', 'Batch size', 16, 8192, true),
      learning_rate: inputNumber('grpo-learning-rate', 'Policy learning rate', 0.000001, 0.01),
      clip_ratio: inputNumber('grpo-clip', 'Clip ratio', 0.01, 0.5),
      entropy_coef: inputNumber('grpo-entropy', 'Entropy weight', 0, 1),
      bc_coef: inputNumber('grpo-bc-weight', 'BC anchor weight', 0, 10),
      target_kl: inputNumber('grpo-kl', 'Target KL', 0.001, 0.5),
      eval_interval: inputNumber('grpo-eval-interval', 'Evaluation interval', 1, 1000, true),
      eval_hands: 100,
      eval_seeds: inputNumber('grpo-eval-seeds', 'Evaluation seeds', 1, 20, true),
      seed: inputNumber('grpo-seed', 'Random seed', 0, 2147483647, true), device: $('grpo-device').value,
      milestone_settings: milestones
    };
    settings.reward_mode = 'game_placement';
    settings.placement_weight = inputNumber('grpo-placement-weight', 'Placement weight', 50, 100) / 100;
    settings.chip_scale_bb100 = 100;
    const groupEpisodes = settings.group_size * 100;
    if (settings.target_episodes % groupEpisodes || settings.rollout_episodes % groupEpisodes) {
      throw new Error(`Training episodes and rollout episodes must be multiples of ${number(groupEpisodes)} (${settings.group_size} attempts × 100 hands). Each complete group counts toward the episode budget.`);
    }
    if (milestones.enabled && milestones.interval % groupEpisodes) {
      throw new Error(`The milestone interval must be a multiple of ${number(groupEpisodes)} so that complete groups finish before saving.`);
    }
    return settings;
  }

  // Keep form choices stable while checkpoint and training progress refresh.
  function renderSources() {
    const checkpoints = list(state.checkpoints), signature = JSON.stringify(checkpoints);
    if (signature !== checkpointSignature) {
      checkpointSignature = signature;
      const selected = $('grpo-source').value;
      const options = [new Option('Choose a completed BC checkpoint', '')];
      for (const checkpoint of checkpoints) {
        const eligible = checkpoint.eligible && checkpoint.status === 'completed';
        const reason = eligible ? `best epoch ${number(checkpoint.epoch)} · completed` : checkpoint.reason || `${checkpoint.status || 'unavailable'} · finish BC training first`;
        const option = new Option(`${checkpoint.name || checkpoint.id} · ${reason}`, String(checkpoint.id));
        option.disabled = !eligible; options.push(option);
      }
      if (!checkpoints.length) options[0] = new Option('No completed BC checkpoints yet', '');
      $('grpo-source').replaceChildren(...options);
      if (checkpoints.some(checkpoint => String(checkpoint.id) === selected)) $('grpo-source').value = selected;
    }
    updateSourceDetail();
  }
  function updateSourceDetail() {
    const checkpoint = sourceCheckpoint();
    if (!checkpoint) { message('grpo-source-detail', 'Choose a completed BC checkpoint. Running BC jobs become available after they finish.'); return; }
    if (!checkpoint.eligible || checkpoint.status !== 'completed') {
      message('grpo-source-detail', checkpoint.reason || 'This BC checkpoint is not ready for GRPO training.', true); return;
    }
    message('grpo-source-detail', `${checkpoint.name || checkpoint.id} · ${number(checkpoint.records)} training decisions · ${checkpoint.device || 'device recorded with source'}. GRPO starts from its best validation snapshot and saves a separate agent.`);
  }
  function renderOpponents() {
    const agents = list(state.agents), available = new Set(agents.map(agent => String(agent.id)));
    for (const id of opponents) if (!available.has(id)) opponents.delete(id);
    const signature = JSON.stringify(agents.map(agent => [agent.id, agent.name, agent.hash]));
    if (signature === opponentSignature) return;
    opponentSignature = signature;
    $('grpo-opponent-list').replaceChildren();
    if (!agents.length) empty($('grpo-opponent-list'), 'Register opponents first', 'Your registered arena bots will appear here.');
    for (const agent of agents) {
      const row = element('label', null, 'lab-check-row'), input = element('input'), info = element('span', null, 'agent-info');
      input.type = 'checkbox'; input.value = String(agent.id); input.checked = opponents.has(input.value);
      input.addEventListener('change', () => { input.checked ? opponents.add(input.value) : opponents.delete(input.value); updateEstimate(); });
      info.append(element('strong', agent.name || agent.id), element('small', `Version ${String(agent.hash || agent.id).slice(0, 12)}`));
      row.append(input, info); $('grpo-opponent-list').append(row);
    }
  }
  function updateEstimate() {
    const enabled = $('grpo-milestones-enabled').checked;
    $('grpo-milestone-inputs').querySelectorAll('input').forEach(input => { input.disabled = !enabled; });
    rewardFields();
    try {
      const settings = payload();
      const groups = settings.target_episodes / (settings.group_size * 100);
      const rollouts = Math.ceil(settings.target_episodes / settings.rollout_episodes);
      const evaluationHands = settings.eval_seeds * settings.table_sizes.reduce((total, size) => total + size, 0) * 100;
      const milestones = settings.milestone_settings;
      const milestoneHands = milestones.eval_seeds * (4 + 5 + 6) * 100;
      const milestoneNote = milestones.enabled
        ? ` Milestones every ${number(milestones.interval)} training episodes evaluate 4/5/6 players: ${number(milestoneHands)} additional table hands per snapshot. Baseline and final milestone evaluations also take time.`
        : ' Milestone evaluations are off.';
      message('grpo-estimate', `${number(settings.target_episodes)} training episodes = ${number(groups)} matched groups, each with ${number(settings.group_size)} attempts. About ${number(rollouts)} rollouts · ${settings.table_sizes.join(', ')} players · ${number(settings.opponents.length)} opponents. Fixed validation runs every ${number(settings.eval_interval)} rollouts (${number(evaluationHands)} additional table hands per evaluation), plus baseline and held-out test evaluations.${milestoneNote} Evaluation hands are not included in the training episode target.`);
    } catch (error) { message('grpo-estimate', error.message, true); }
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
  function bb100(evaluation) {
    if (typeof evaluation === 'number') return evaluation;
    if (!evaluation) return null;
    const overall = evaluation.overall || evaluation;
    const value = overall.bb100 ?? overall.bb_per_100 ?? overall.overall_bb100;
    return finite(value) ? Number(value) : null;
  }
  function baselineEvaluation(job) { return job.baseline?.validation || job.baseline; }
  function placementJob(job) {
    return true;
  }
  function scoreCard(label, evaluation, note, placement = false) {
    const value = placement ? (finite(evaluation?.selection_score) ? Number(evaluation.selection_score) : null) : bb100(evaluation);
    const card = element('div', null, 'ppo-score');
    const unit = placement ? ' round score' : ' bb/100';
    card.append(element('span', label), element('strong', value == null ? '—' : `${number(value)}${unit}`, value == null ? '' : value < 0 ? 'negative' : 'positive'));
    if (placement && evaluation) card.append(element('small', `${number(bb100(evaluation))} bb/100 · normalized round placement ${number(evaluation.normalized_round_placement)}`));
    card.append(element('small', note)); return card;
  }
  function metricForSize(evaluation, size) {
    const groups = evaluation?.by_table_size || evaluation?.overall?.by_table_size;
    if (Array.isArray(groups)) return groups.find(group => Number(group.table_size ?? group.size) === size);
    return groups?.[size];
  }
  function progressText(job) {
    if (typeof job.progress === 'string') return job.progress;
    return job.progress?.message || job.progress?.progress || '';
  }
  function jobKey(id, source = 'local') { return source === 'live' ? `live:${id}` : String(id); }
  function actionButton(id, action, text, source = 'local') {
    const button = element('button', text, 'button secondary'); button.type = 'button';
    button.dataset.grpoId = String(id); button.dataset.grpoAction = action; button.dataset.grpoSource = source;
    button.disabled = pending.has(jobKey(id, source));
    return button;
  }
  function snapshotButtons(job, source = 'local', milestone = null) {
    const tools = element('div', null, 'run-tools');
    const label = milestone ? 'milestone' : 'training-best';
    for (const [format, text] of [['zip', 'agent ZIP'], ['checkpoint', 'checkpoint']]) {
      const button = actionButton(job.id, 'snapshot', `Download ${label} ${text}`, source);
      button.dataset.grpoKind = milestone ? 'milestone' : 'best';
      button.dataset.grpoFormat = format;
      if (milestone) button.dataset.grpoMilestone = milestone.id;
      tools.append(button);
    }
    return tools;
  }
  function trainingSnapshot(job, source = 'local') {
    const box = element('section', null, 'ppo-snapshot-panel');
    box.append(element('h3', 'Training-best downloads'));
    const best = job.snapshot?.best;
    if (best?.available || (source === 'live' && job.best_available)) {
      const metric = placementJob(job) ? 'weighted duplicate-round placement and chip score' : 'bb/100';
      const selection = source === 'live' && !job.selection_metric && !best?.selection_metric
        ? 'Uses the saved checkpoint selected by that job’s validation objective.'
        : `Uses the saved checkpoint selected by validation ${metric}.`;
      box.append(element('p', `${selection} Its objective, exact episode count and iteration are shown after the copy is prepared.`, 'small muted'));
      box.append(snapshotButtons(job, source));
    } else {
      box.append(element('p', best?.reason || (best ? 'A validated training-best checkpoint is not available yet.' : 'Saved-checkpoint download details are unavailable for this job.'), 'small muted'));
    }
    return box;
  }
  function downloadLink(url, text) {
    // Downloads must remain on this dashboard; API text is never inserted as HTML.
    if (typeof url !== 'string' || !url.startsWith('/api/') || url.startsWith('//')) return null;
    const link = element('a', text, 'button secondary'); link.href = url; link.download = '';
    return link;
  }
  function actionStatus(job, source = 'local') {
    const action = actionMessages.get(jobKey(job.id, source));
    const status = element('div', null, 'grpo-action-status');
    status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite'); status.hidden = !action;
    if (!action) return status;
    status.append(element('p', action.text, action.error ? 'small error-text' : 'small'));
    if (action.snapshot) {
      const result = action.snapshot, metadata = result.metadata || result;
      const episodes = result.episodes ?? metadata.episodes;
      const iteration = result.iteration ?? metadata.iteration;
      const label = result.kind === 'milestone' ? 'Milestone snapshot' : 'Training-best snapshot';
      status.append(element('p', `${label} · ${number(episodes)} training episodes · iteration ${number(iteration)}. These links keep this saved version even if training advances.`, 'small muted'));
      if (result.selection_metric) status.append(element('p', `Selection metric: ${result.selection_metric === 'placement_chip_score' ? 'weighted duplicate-round placement and chip score' : result.selection_metric}.`, 'small muted'));
      if (result.kind !== 'milestone' && finite(iteration) && Number(iteration) === 0) status.append(element('p', 'Iteration 0 is the initial BC policy.', 'small muted'));
      const links = element('div', null, 'run-tools');
      for (const [url, title] of [[result.zip_url, 'Agent ZIP'], [result.checkpoint_url, 'PyTorch checkpoint']]) {
        const link = downloadLink(url, title); if (link) links.append(link);
      }
      status.append(links);
    }
    return status;
  }
  function milestoneTitle(snapshot) {
    const kind = snapshot.kind === 'baseline' ? 'Initial BC' : snapshot.kind === 'final' ? 'Final policy' : 'Milestone';
    return `${kind} · ${number(snapshot.episodes)} episodes`;
  }
  function milestoneMetrics(snapshot) {
    return table([4, 5, 6].map(size => {
      const row = snapshot.by_table_size?.[size] || {};
      return [size, number(row.mean_rank), number(row.mean_points), number(row.normalized_placement),
        number(row.net_chips), number(row.bb100), number(row.learner_failures), number(row.opponent_failures),
        `${number(row.completed_games)} / ${number(row.attempted_games)}`,
        `${number(row.completed_hands)} / ${number(row.attempted_hands)}`];
    }), ['Players', 'Avg. game placement ↓', 'Avg. game points ↑', 'Normalized game placement ↑', 'Net chips', 'bb/100',
      'Learner failures', 'Opponent failures', 'Games completed / attempted', 'Hands completed / attempted']);
  }
  function renderMilestones(job) {
    const section = element('section', null, 'ppo-milestones');
    section.append(element('h3', 'Milestone evaluations'));
    const milestones = job.milestones;
    if (!milestones) {
      section.append(element('p', 'No milestone history is recorded for this job. Training validation remains available above.', 'small muted'));
      return section;
    }
    const snapshots = [milestones.baseline, ...list(milestones.snapshots)].filter(Boolean);
    const settings = milestones.settings || {};
    const tracking = finite(milestones.tracking_start_episodes) ? ` Tracking started at ${number(milestones.tracking_start_episodes)} episodes.` : '';
    section.append(element('p', `${milestones.enabled ? `Every ${number(settings.interval)} episodes · ${number(settings.eval_hands)} hands per game · ${number(settings.eval_seeds)} seed groups.` : 'Milestone tracking is off.'}${tracking} Each result shows the actual saved episode count.`, 'small muted'));
    const missingMilestones = list(milestones.unrecoverable_past_milestones);
    if (missingMilestones.length) section.append(element('p', `${number(missingMilestones.length)} earlier milestone checkpoint${missingMilestones.length === 1 ? ' was' : 's were'} not saved and cannot be reconstructed. Tracking begins from the saved progress shown above.`, 'small muted'));
    if (milestones.error) section.append(element('p', String(milestones.error), 'notice error'));
    if (list(milestones.pending).length) section.append(element('p', `${number(milestones.pending.length)} saved evaluation${milestones.pending.length === 1 ? '' : 's'} pending.`, 'small muted'));
    if (!snapshots.length) {
      section.append(element('p', 'No saved milestone evaluations yet.', 'small muted'));
      return section;
    }
    const leaders = element('div', null, 'ppo-milestone-leaders');
    for (const [metric, label] of [['bb100', 'Milestone best · bb/100'], ['normalized_placement', 'Milestone best · game placement']]) {
      const selected = snapshots.find(snapshot => snapshot.id === milestones.leaders?.[metric]);
      const card = element('div', null, 'ppo-score');
      card.append(element('span', label), element('strong', selected ? number(selected[metric]) : '—'));
      card.append(element('small', selected ? milestoneTitle(selected) : 'Awaiting an eligible completed evaluation'));
      if (selected?.checkpoint_available) card.append(snapshotButtons(job, 'local', selected));
      leaders.append(card);
    }
    section.append(leaders, element('p', 'Milestone leaders use these separate evaluations. Training best follows the objective saved with that job.', 'small muted'));
    if (milestones.round_leaders) {
      const roundLeaders = element('div', null, 'ppo-milestone-leaders');
      for (const [metric, label] of [['placement_chip_score', 'Milestone best · weighted round score'], ['normalized_round_placement', 'Milestone best · round placement']]) {
        const selected = snapshots.find(snapshot => snapshot.id === milestones.round_leaders[metric]);
        const result = selected?.round_metrics || selected;
        const value = metric === 'placement_chip_score' ? result?.selection_score : result?.normalized_round_placement;
        const card = element('div', null, 'ppo-score');
        card.append(element('span', label), element('strong', number(value)), element('small', selected ? milestoneTitle(selected) : 'Awaiting complete duplicate rounds'));
        if (selected?.checkpoint_available) card.append(snapshotButtons(job, 'local', selected));
        roundLeaders.append(card);
      }
      section.append(roundLeaders);
    }
    section.append(element('p', 'Placement 1 is best. Game points average the points earned per game, with ties shared. Normalized placement runs from 0 (last) to 1 (first); overall milestone scores weight 4-, 5- and 6-player results equally. Failures count bot verdicts per game. Chip rates score full completed games; partial hands appear only in coverage.', 'small muted'));
    for (const snapshot of snapshots) {
      const details = element('details', null, 'raw ppo-milestone');
      details.dataset.grpoDetails = `milestone:${snapshot.id}`;
      const summary = element('summary'), title = element('span', milestoneTitle(snapshot));
      const status = snapshot.status || 'unknown';
      const badge = element('span', status, 'tag');
      if (['pending', 'completed', 'failed'].includes(status)) badge.classList.add(`status-${status}`);
      summary.append(title, badge); details.append(summary);
      details.append(element('p', `Saved iteration ${number(snapshot.iteration)} · ${number(snapshot.completed_games)} completed / ${number(snapshot.expected_games)} planned games · ${number(snapshot.completed_hands)} completed hands`, 'small muted'));
      if (snapshot.error) details.append(element('p', String(snapshot.error), 'notice error'));
      details.append(milestoneMetrics(snapshot));
      const rounds = snapshot.round_metrics || (snapshot.selection_metric === 'placement_chip_score' ? snapshot : null);
      if (rounds) {
        details.append(element('h3', 'Duplicate-round scores'));
        details.append(table([4, 5, 6].map(size => {
          const row = rounds.by_table_size?.[size] || {};
          return [size, number(row.mean_round_points), number(row.normalized_round_placement), number(row.selection_score), `${number(row.completed_rounds)} / ${number(row.expected_rounds)}`];
        }), ['Players', 'Avg. round points ↑', 'Normalized round placement ↑', 'Weighted round score ↑', 'Complete / planned rounds']));
      }
      if (snapshot.status === 'failed' || (snapshot.status === 'completed' && snapshot.eligible_for_best === false)) details.append(element('p', 'This evaluation is excluded from milestone-best selection. Review its failure counts and coverage.', 'small muted'));
      if (snapshot.resumable === false) details.append(element('p', 'This is a BC checkpoint. It supplies the starting policy rather than a resumable GRPO iteration.', 'small muted'));
      if (snapshot.checkpoint_available) details.append(snapshotButtons(job, 'local', snapshot));
      section.append(details);
    }
    return section;
  }
  function evaluationPlot(job) {
    const history = list(job.history);
    const placement = placementJob(job);
    const validationKey = placement ? 'validation_score' : 'validation_bb100';
    const trainingKey = placement ? 'train_reward' : 'train_bb100';
    if (!history.some(row => finite(row[validationKey]) || finite(row[trainingKey]))) return null;
    const baseline = baselineEvaluation(job);
    const initial = placement ? (finite(baseline?.selection_score) ? Number(baseline.selection_score) : null) : bb100(baseline);
    const values = [];
    for (const row of history) {
      if (finite(row[validationKey])) values.push(Number(row[validationKey]));
      if (finite(row[trainingKey])) values.push(Number(row[trainingKey]));
    }
    if (initial != null) values.push(initial);
    if (!values.length) return null;
    const minimumValue = Math.min(...values, 0), maximumValue = Math.max(...values, 0);
    const padding = Math.max((maximumValue - minimumValue) * 0.12, placement ? 0.05 : 1);
    const minimum = minimumValue - padding, maximum = maximumValue + padding;
    const lastEpisode = Math.max(1, ...history.map(row => Number(row.episodes || 0)));
    const namespace = 'http://www.w3.org/2000/svg', frame = element('div', null, 'lab-chart ppo-chart');
    const svg = document.createElementNS(namespace, 'svg');
    svg.setAttribute('viewBox', '0 0 640 230'); svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', placement ? 'GRPO game-placement reward and duplicate-round validation score, compared with the BC baseline.' : 'Poker results during GRPO: validation and training big blinds per 100 hands, compared with the initial BC validation baseline.');
    function svgElement(tag, attributes, text) {
      const node = document.createElementNS(namespace, tag);
      for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
      if (text != null) node.textContent = String(text);
      return node;
    }
    const x = episode => 64 + Number(episode) / lastEpisode * 558;
    const y = value => 188 - (Number(value) - minimum) / (maximum - minimum) * 162;
    for (let step = 0; step <= 4; step++) {
      const value = minimum + (maximum - minimum) * step / 4;
      svg.append(svgElement('line', { x1: 64, x2: 622, y1: y(value), y2: y(value), class: 'lab-chart-grid' }));
      svg.append(svgElement('text', { x: 3, y: y(value) + 4 }, number(value)));
    }
    if (initial != null) {
      svg.append(svgElement('line', { x1: 64, x2: 622, y1: y(initial), y2: y(initial), stroke: '#ebc17c', 'stroke-width': 1.8, 'stroke-dasharray': '6 4' }));
    }
    for (const [key, color, opacity] of [[trainingKey, '#8cbaff', 0.6], [validationKey, '#55e0b0', 1]]) {
      const points = history.filter(row => finite(row[key])).map(row => [x(row.episodes || 0), y(row[key])]);
      if (key === validationKey && initial != null && !history.some(row => Number(row.episodes) === 0 && finite(row[key]))) points.unshift([x(0), y(initial)]);
      if (!points.length) continue;
      svg.append(svgElement('polyline', { points: points.map(point => point.join(',')).join(' '), fill: 'none', stroke: color, 'stroke-width': key === validationKey ? 2.8 : 1.5, opacity }));
      for (const point of points) svg.append(svgElement('circle', { cx: point[0], cy: point[1], r: key === validationKey ? 3 : 2, fill: color, opacity }));
    }
    svg.append(svgElement('text', { x: 64, y: 211 }, '0 episodes'), svgElement('text', { x: 622, y: 211, 'text-anchor': 'end' }, `${number(lastEpisode)} episodes`));
    svg.append(svgElement('text', { x: 64, y: 14 }, placement ? 'Normalized score' : 'bb/100'));
    frame.append(svg);
    const legend = element('div', null, 'ppo-chart-legend');
    legend.append(element('span', placement ? 'Validation · round score' : 'Validation', 'ppo-legend-validation'), element('span', placement ? 'Training · game reward' : 'Training returns', 'ppo-legend-training'), element('span', 'Initial BC validation', 'ppo-legend-baseline'));
    frame.append(legend, element('p', placement ? 'Training rewards game placement; validation scores duplicate-round placement. Both are scaled from −1 to 1 with a bounded chip component. The round-validation score selects the checkpoint; these are different measurements.' : 'Higher bb/100 is better. Training returns use changing policy data; fixed validation games choose the saved policy. The held-out test is shown separately.', 'small muted'));
    return frame;
  }
  function groupDiagnostics(job) {
    const history = list(job.history);
    if (!history.some(row => finite(row.reward_std) || finite(row.zero_spread_fraction))) return null;
    const section = element('section', null, 'grpo-group-diagnostics');
    section.append(element('h3', 'Is the group providing a learning signal?'));
    section.append(element('p', 'Reward spread shows how much the attempts differed. Groups with no reward spread contribute zero relative reward advantage. A high spread is not a measure of poker strength; compare fixed validation and held-out test results above.', 'small muted'));
    const charts = element('div', null, 'grpo-group-charts');
    for (const [key, label, scale, color] of [
      ['reward_std', 'Mean reward spread within groups', 1, '#8cbaff'],
      ['zero_spread_fraction', 'Groups with no reward spread (%)', 100, '#ebc17c']
    ]) {
      const rows = history.filter(row => finite(row[key]));
      if (!rows.length) continue;
      const frame = element('div', null, 'lab-chart ppo-chart');
      const namespace = 'http://www.w3.org/2000/svg';
      const svg = document.createElementNS(namespace, 'svg');
      svg.setAttribute('viewBox', '0 0 460 190');
      svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', label);
      function shape(tag, attributes, text) {
        const node = document.createElementNS(namespace, tag);
        for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, String(value));
        if (text != null) node.textContent = String(text);
        return node;
      }
      const lastEpisode = Math.max(1, ...rows.map(row => Number(row.episodes || 0)));
      const maximum = scale === 100 ? 100 : Math.max(0.01, ...rows.map(row => Number(row[key]) * scale)) * 1.1;
      const x = episode => 48 + Number(episode) / lastEpisode * 390;
      const y = value => 147 - Number(value) / maximum * 115;
      for (let step = 0; step <= 4; step++) {
        const value = maximum * step / 4;
        svg.append(shape('line', { x1: 48, x2: 438, y1: y(value), y2: y(value), class: 'lab-chart-grid' }));
        svg.append(shape('text', { x: 2, y: y(value) + 4 }, scale === 100 ? number(value) : precise(value)));
      }
      const points = rows.map(row => [x(row.episodes || 0), y(Number(row[key]) * scale)]);
      svg.append(shape('polyline', { points: points.map(point => point.join(',')).join(' '), fill: 'none', stroke: color, 'stroke-width': 2 }));
      for (const point of points) svg.append(shape('circle', { cx: point[0], cy: point[1], r: 2.5, fill: color }));
      svg.append(shape('text', { x: 48, y: 171 }, '0 episodes'));
      svg.append(shape('text', { x: 438, y: 171, 'text-anchor': 'end' }, `${number(lastEpisode)} episodes`));
      frame.append(element('strong', label, 'small'), svg);
      const latest = rows.at(-1);
      frame.append(element('p', `Latest: ${scale === 100 ? number(Number(latest[key]) * scale) + '%' : precise(latest[key])}`, 'small muted'));
      charts.append(frame);
    }
    section.append(charts);
    const latest = history.at(-1);
    if (latest?.update_skipped) section.append(element('p', 'The latest rollout skipped its policy update because there was no usable relative reward signal.', 'notice'));
    return section;
  }
  function renderJob(node, job) {
    node.replaceChildren();
    const config = job.config || {}, history = list(job.history), latest = history.at(-1) || {};
    const placement = placementJob(job);
    const heading = element('div', null, 'lab-card-heading'), title = element('div', null, 'run-title');
    title.append(element('strong', job.name || job.id), element('small', `Created ${date(job.created)} · ${job.source_name || config.source_name || 'From a saved BC checkpoint'}`));
    const status = job.pause_requested && job.status === 'running' ? 'Pausing…' : job.status || 'unknown';
    const badge = element('span', status, 'tag');
    if (/^(queued|running|paused|interrupted|failed|completed)$/.test(job.status || '')) badge.classList.add(`status-${job.status}`);
    heading.append(title, badge); node.append(heading);
    const target = Number(job.target_episodes || config.target_episodes || 1), episodes = Number(job.episodes || 0);
    const progress = element('div', null, 'lab-progress'), bar = element('progress');
    bar.max = Math.max(1, target); bar.value = Math.max(0, Math.min(target, episodes)); bar.setAttribute('aria-label', 'Completed GRPO training episodes');
    progress.append(bar, element('span', `${number(episodes)} / ${number(target)} training episodes · ${Math.min(100, episodes / target * 100).toFixed(1)}%`, 'small muted')); node.append(progress);
    const facts = element('div', null, 'lab-facts');
    facts.append(element('span', `Iteration ${number(job.iteration || 0)}`), element('span', `${number(job.decisions || 0)} learner decisions`));
    if (job.device || config.device) facts.append(element('span', `Device: ${job.device || config.device}`));
    if (finite(job.elapsed_seconds)) facts.append(element('span', `Elapsed ${duration(job.elapsed_seconds)}`));
    facts.append(element('span', `${number(config.group_size || 4)} attempts per matched group`));
    if (placement && finite(latest.train_reward)) facts.append(element('span', `Latest game reward: ${number(latest.train_reward)}`));
    if (finite(latest.train_bb100)) facts.append(element('span', `${placement ? 'Latest chip profit' : 'Latest training return'}: ${number(latest.train_bb100)} bb/100`));
    node.append(facts);
    const rewardLabel = placement ? `${number(Number(config.placement_weight ?? 0.9) * 100)}% game placement / ${number((1 - Number(config.placement_weight ?? 0.9)) * 100)}% bounded chips · 100-hand games · public game context` : 'Net chip profit per hand · legacy reward';
    node.append(element('p', `Reward: ${rewardLabel}.`, 'notice'));
    const text = [job.phase, progressText(job)].filter(Boolean).join(' · ');
    if (text) node.append(element('p', text, 'small muted'));
    if (job.error) node.append(element('p', job.error, 'notice error'));
    const tools = element('div', null, 'run-tools');
    if (['queued', 'running'].includes(job.status) && !job.pause_requested) tools.append(actionButton(job.id, 'pause', 'Pause GRPO'));
    if (['paused', 'interrupted', 'failed'].includes(job.status)) tools.append(actionButton(job.id, 'resume', 'Resume GRPO'));
    if (job.status === 'completed' && !job.agent_id) tools.append(actionButton(job.id, 'export', 'Create GRPO agent'));
    if (job.agent_id) {
      const download = element('a', 'Download agent ZIP', 'button secondary'); download.href = `/api/agents/${encodeURIComponent(job.agent_id)}/download`; download.download = '';
      const launch = element('a', 'Test in arena →', 'button primary'); launch.href = '#launch'; tools.append(download, launch);
      node.append(element('p', 'The selected policy is registered as a separate agent. Choose it and your opponents in the arena, then test it with fresh seeds.', 'notice'));
    }
    const folders = job.folders || {};
    if (state.folder_actions_available === true && state.finder_available === true) {
      for (const folder of ['agent', 'checkpoints']) {
        if (!folders[folder] || (folder === 'agent' && !job.agent_id)) continue;
        const button = actionButton(job.id, 'open-folder', `Open ${folder} in Finder`);
        button.dataset.grpoFolder = folder; tools.append(button);
      }
    }
    if (tools.childNodes.length) node.append(tools);
    for (const [folder, label] of [['agent', 'Agent folder'], ['checkpoints', 'Checkpoints folder']]) {
      if (!folders[folder]) continue;
      const path = element('p', `${label}: `, 'small muted lab-path');
      path.append(element('code', folders[folder])); node.append(path);
    }
    if (state.folder_actions_available !== true) node.append(element('p', 'Folder controls will be available after the dashboard server restarts.', 'small muted'));
    const scores = element('div', null, 'ppo-score-grid');
    const savedValidation = job.best_validation || (job.status === 'completed' ? job.validation : null);
    const bestScore = savedValidation || (finite(job.best_score) ? Number(job.best_score) : null);
    scores.append(
      scoreCard('Initial BC · validation', baselineEvaluation(job), 'Starting policy on fixed validation games', placement),
      scoreCard('Training best · validation', bestScore, placement ? 'Selected by weighted duplicate-round placement and chips' : 'Selected by validation bb/100', placement),
      scoreCard('Initial BC · held-out test', job.baseline_test, 'Starting policy on separate test games', placement),
      scoreCard('Training best · held-out test', job.test, 'Selected policy on the same test games', placement)
    );
    node.append(scores);
    if (job.best_iteration != null) {
      node.append(element('p', Number(job.best_iteration) === 0 ? 'Training best: initial BC checkpoint (iteration 0).' : `Training best: GRPO iteration ${number(job.best_iteration)}.`, 'small ppo-best-policy'));
    }
    const chart = evaluationPlot(job); if (chart) node.append(chart);
    const groupCharts = groupDiagnostics(job); if (groupCharts) node.append(groupCharts);
    node.append(element('p', 'Training best may be the original BC policy if GRPO does not improve validation. GRPO loss measures the update objective, not playing strength. Relative learning needs different rewards within a group; repeated equal results provide no reward advantage.', 'small muted'));
    node.append(trainingSnapshot(job), renderMilestones(job));
    const details = element('details', null, 'raw'); details.append(element('summary', 'Results by table size, update diagnostics and settings'));
    details.dataset.grpoDetails = 'training';
    const sizes = list(config.table_sizes).length ? config.table_sizes : [4, 5, 6];
    const displayedValidation = savedValidation || job.validation;
    const validationLabel = savedValidation ? 'Training-best validation · bb/100' : 'Latest validation · bb/100';
    const sizeRows = sizes.map(size => [size, number(bb100(metricForSize(baselineEvaluation(job), size))), number(bb100(metricForSize(displayedValidation, size))), number(bb100(metricForSize(job.baseline_test, size))), number(bb100(metricForSize(job.test, size)))]);
    details.append(table(sizeRows, ['Players', 'Initial BC validation · bb/100', validationLabel, 'Initial BC test · bb/100', 'Training-best test · bb/100']));
    if (placement) {
      details.append(element('h3', 'Round-placement comparison'));
      details.append(table(sizes.map(size => {
        const original = metricForSize(baselineEvaluation(job), size) || {};
        const selected = metricForSize(displayedValidation, size) || {};
        const originalTest = metricForSize(job.baseline_test, size) || {};
        const selectedTest = metricForSize(job.test, size) || {};
        return [size, number(original.selection_score), number(selected.selection_score), number(originalTest.selection_score), number(selectedTest.selection_score), number(selectedTest.normalized_round_placement)];
      }), ['Players', 'BC validation score', 'Best validation score', 'BC test score', 'Best test score', 'Best test · normalized round placement']));
    }
    if (job.test || job.baseline_test) details.append(element('p', 'Initial BC and the selected policy use identical held-out test seeds. These test games are separate from validation and do not choose the saved policy.', 'small muted'));
    if (history.length) {
      details.append(element('h3', 'Update diagnostics', 'lab-subheading'));
      details.append(element('p', 'KL tracks policy change; entropy tracks action uncertainty. These diagnostics help inspect learning but do not establish poker strength.', 'small muted'));
      details.append(table(history.map(row => [row.iteration, number(row.episodes), number(row.train_bb100), number(row.validation_bb100), precise(row.policy_loss), precise(row.entropy), precise(row.approx_kl), finite(row.clip_fraction) ? `${number(Number(row.clip_fraction) * 100)}%` : '—', precise(row.bc_kl), number(row.updates), row.early_stopped == null ? '—' : row.early_stopped ? 'Yes' : 'No', row.update_skipped == null ? '—' : row.update_skipped ? 'Yes' : 'No']), ['Iteration', 'Episodes', 'Train bb/100', 'Validation bb/100', 'Policy loss', 'Entropy', 'Approx. KL', 'Clipped', 'BC KL', 'Updates', 'KL early stop', 'Update skipped']));
      details.append(table(history.map(row => [row.iteration, number(row.groups), precise(row.reward_std), finite(row.zero_spread_fraction) ? `${number(Number(row.zero_spread_fraction) * 100)}%` : '—']), ['Iteration', 'Groups in rollout', 'Mean group reward spread (std. dev.)', 'Groups with no reward spread']));
      if (placement) details.append(table(history.map(row => [row.iteration, number(row.episodes), number(row.train_reward), number(row.validation_score)]), ['Iteration', 'Episodes', 'Training · game reward', 'Validation · round score']));
    }
    details.append(element('h3', 'Evaluation details', 'lab-subheading'), element('pre', JSON.stringify({ baseline: job.baseline || null, latest_validation: job.latest_validation || job.validation || null, best_validation: savedValidation, best_score: job.best_score ?? null, baseline_test: job.baseline_test || null, test: job.test || null }, null, 2)));
    details.append(element('h3', 'Training settings', 'lab-subheading'), element('pre', JSON.stringify(config, null, 2)));
    node.append(details);
    node.append(actionStatus(job));
  }
  function renderJobs() {
    const jobs = list(state.grpo_jobs), ids = new Set(jobs.map(job => String(job.id))), parent = $('grpo-jobs');
    $('grpo-job-count').textContent = `${jobs.length} GRPO job${jobs.length === 1 ? '' : 's'}`;
    for (const [id, card] of jobCards) if (!ids.has(id)) { card.node.remove(); jobCards.delete(id); }
    if (!jobs.length) { empty(parent, 'Continue learning from your BC student', 'Choose a completed checkpoint and opponents, then start a GRPO experiment.'); return; }
    if (parent.querySelector(':scope > .empty')) parent.replaceChildren();
    for (const job of jobs) {
      const id = String(job.id), signature = JSON.stringify([job, state.folder_actions_available, state.finder_available, pending.has(id), actionMessages.get(id)]);
      let card = jobCards.get(id);
      if (!card) { card = { node: element('article', null, 'lab-card grpo-job'), signature: '' }; jobCards.set(id, card); parent.append(card.node); }
      if (card.signature === signature) continue;
      const opened = new Set(Array.from(card.node.querySelectorAll('details[open][data-grpo-details]'), details => details.dataset.grpoDetails));
      renderJob(card.node, job);
      for (const details of card.node.querySelectorAll('details[data-grpo-details]')) details.open = opened.has(details.dataset.grpoDetails);
      card.signature = signature;
    }
    jobs.forEach((job, index) => {
      const node = jobCards.get(String(job.id)).node;
      if (parent.children[index] !== node) parent.insertBefore(node, parent.children[index] || null);
    });
  }
  async function refresh() {
    if (refreshing) return;
    refreshing = true;
    try {
      state = await api('/api/grpo');
      window.dispatchEvent(new CustomEvent('arena-state', { detail: { source: 'grpo', state } }));
      const development = state.development_mode === true;
      $('grpo-development-notice').hidden = !development;
      $('grpo-device-note').hidden = !development;
      $('grpo-device').querySelector('option[value="mps"]').disabled = development;
      $('grpo-device').querySelector('option[value="cuda"]').disabled = development;
      $('grpo-device').querySelector('option[value="auto"]').textContent = development ? 'Automatic · CPU in this test workspace' : 'Automatic · detect available GPU';
      if (development) $('grpo-device').value = 'cpu';
      renderSources(); renderOpponents(); updateEstimate(); renderJobs(); $('grpo-error').hidden = true;
    } catch (error) {
      $('grpo-error').textContent = `GRPO status unavailable: ${error.message} Retrying automatically.`; $('grpo-error').hidden = false;
    } finally { refreshing = false; }
  }

  $('grpo-source').addEventListener('change', () => { updateSourceDetail(); updateEstimate(); });
  $('grpo-opponents-all').onclick = () => {
    for (const agent of list(state.agents)) opponents.add(String(agent.id));
    $('grpo-opponent-list').querySelectorAll('input').forEach(input => { input.checked = true; }); updateEstimate();
  };
  $('grpo-opponents-none').onclick = () => {
    opponents.clear(); $('grpo-opponent-list').querySelectorAll('input').forEach(input => { input.checked = false; }); updateEstimate();
  };
  $('grpo-form').addEventListener('input', updateEstimate);
  $('grpo-form').addEventListener('change', updateEstimate);
  $('grpo-form').addEventListener('submit', async event => {
    event.preventDefault(); $('grpo-create').disabled = true;
    try {
      const settings = payload();
      message('grpo-status', 'Saving GRPO experiment…');
      await api('/api/grpo', settings);
      message('grpo-status', 'GRPO training queued. Follow its progress below; the BC source checkpoint is preserved.'); await refresh();
    } catch (error) { message('grpo-status', error.message, true); }
    finally { $('grpo-create').disabled = false; }
  });
  async function handleJobAction(event) {
    const button = event.target.closest('button[data-grpo-action]');
    if (!button || button.disabled) return;
    const id = button.dataset.grpoId, action = button.dataset.grpoAction, folder = button.dataset.grpoFolder;
    const source = button.dataset.grpoSource || 'local', key = jobKey(id, source);
    if (source !== 'local') return;
    if (pending.has(key)) return;
    pending.add(key); button.disabled = true;
    function actionMessage(text, error = false, snapshot = null) {
      actionMessages.set(key, { text, error, snapshot }); message('grpo-status', text, error);
      const status = button.closest('.grpo-job')?.querySelector('.grpo-action-status');
      if (status) { status.textContent = text; status.hidden = false; status.classList.toggle('error-text', error); }
    }
    try {
      if (action === 'snapshot') {
        const kind = button.dataset.grpoKind;
        const body = { kind };
        if (kind === 'milestone') body.milestone_id = button.dataset.grpoMilestone;
        actionMessage(source === 'live' ? 'Copying the saved main-dashboard checkpoint and checking its agent in this preview…' : 'Copying the saved checkpoint and checking its downloadable agent…');
        const prefix = '/api/grpo/';
        const result = await api(`${prefix}${encodeURIComponent(id)}/snapshot`, body);
        const url = button.dataset.grpoFormat === 'checkpoint' ? result.checkpoint_url : result.zip_url;
        const link = downloadLink(url, 'Download saved snapshot');
        if (!link) throw new Error('The snapshot was prepared, but the server did not return a valid download link.');
        actionMessage('Snapshot ready. Your download is starting; both saved files are available below.', false, result);
        link.hidden = true; document.body.append(link); link.click(); link.remove();
      } else if (action === 'open-folder') {
        if (state.folder_actions_available !== true || state.finder_available !== true) throw new Error('Finder controls are unavailable on this dashboard server.');
        actionMessage(`Opening ${folder} folder in Finder…`);
        const result = await api(`/api/grpo/${encodeURIComponent(id)}/open-folder`, { folder });
        actionMessage(`Opened ${folder} folder in Finder: ${result.path}`);
      } else {
        actionMessage(action === 'export' ? 'Packaging and checking the selected policy…' : 'Saving request…');
        await api(`/api/grpo/${encodeURIComponent(id)}/${action}`, {});
        const messages = { pause: 'Pause requested. Completed iterations are preserved; an unfinished iteration is replayed on resume.', resume: 'GRPO queued to resume from its saved checkpoint.', export: 'The selected GRPO policy is registered as a separate agent. Test it in the arena with fresh seeds.' };
        actionMessage(messages[action] || 'Saved.');
      }
    } catch (error) { actionMessage(error.message, true); }
    finally { pending.delete(key); button.disabled = false; await refresh(); renderJobs(); }
  }
  $('grpo-jobs').addEventListener('click', handleJobAction);
  refresh(); setInterval(refresh, 2000);
})();
