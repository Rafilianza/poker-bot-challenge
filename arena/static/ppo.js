/* PPO training workspace. API values are rendered as text, never as HTML. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const list = value => Array.isArray(value) ? value : [];
  const finite = value => value != null && value !== '' && Number.isFinite(Number(value));
  const number = value => finite(value) ? Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 }) : '—';
  const precise = value => finite(value) ? Number(value).toLocaleString(undefined, { maximumFractionDigits: 6 }) : '—';
  const opponents = new Set(), pending = new Set(), jobCards = new Map(), liveCards = new Map(), actionMessages = new Map();
  let state = { checkpoints: [], agents: [], ppo_jobs: [] };
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
      if (error.name === 'AbortError') throw new Error('The request timed out. Check the PPO job list before retrying; the server may have accepted it.');
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
    return Array.from(document.querySelectorAll('input[name="ppo-size"]:checked'), input => Number(input.value));
  }
  function sourceCheckpoint() {
    return list(state.checkpoints).find(checkpoint => String(checkpoint.id) === $('ppo-source').value);
  }
  function trainingModeLabel(mode) {
    return { bots: 'Existing bots', self_play: 'Self-play', mixed: 'Mixed self-play' }[mode] || mode;
  }
  function trainingFields() {
    const mode = $('ppo-training-mode').value;
    const selfPlay = mode !== 'bots', mixed = mode === 'mixed';
    const available = state.self_play_available === true;
    for (const option of $('ppo-training-mode').options) {
      if (option.value !== 'bots') option.disabled = !available;
    }
    $('ppo-self-play-history').hidden = !mixed;
    $('ppo-history-limit').disabled = !mixed;
    $('ppo-history-interval').disabled = !mixed;
    $('ppo-self-play-note').hidden = !selfPlay;
    $('ppo-self-play-measurement').hidden = !selfPlay;
    $('ppo-self-play-note').textContent = mode === 'mixed'
      ? '4 players = learner + 3 opponents; 5 players = learner + 4; 6 players = learner + 5. Current-policy copies refresh after each PPO iteration. Each seat has independent cards, history and random action choices. All opponents remain fixed during a rollout. Only the learner supplies training decisions; its seat rotates.'
      : '4 players = learner + 3 copies; 5 players = learner + 4; 6 players = learner + 5. Copies use identical weights with independent cards, histories and random action choices. They refresh after each PPO iteration and remain fixed during a rollout. Only the learner supplies training decisions; its seat rotates.';
    $('ppo-opponent-heading').textContent = mode === 'self_play' ? 'Benchmark opponents' : 'Training + benchmark opponents';
    const notes = {
      bots: 'Train against the registered bots selected below. The same fixed pool supplies validation opponents.',
      self_play: 'Train only against copies of the current learner. Selected registered bots are required for fixed validation and milestone assessments; they do not enter training games.',
      mixed: 'Each opponent seat draws 50% from current copies, 25% from past versions and 25% from selected registered bots. These are probabilities, so each table can have a different mix. Before older versions exist, historical draws use the current policy.'
    };
    const availabilityNote = available ? '' : ' Self-play requires the updated dashboard server.';
    message('ppo-training-mode-note', `${notes[mode] || 'Choose a training mode.'}${availabilityNote}`, selfPlay && !available);
    const usage = mode === 'self_play'
      ? 'Select registered bots for fixed validation and milestone assessments. Pure self-play uses copies during training.'
      : 'Select registered bots for training and fixed validation.';
    $('ppo-opponent-note').textContent = `${usage} Milestones require at least five distinct opponents for 4-, 5- and 6-player tables. With milestones off, choose enough to fill your selected validation table sizes.`;
  }
  function placementMode() { return $('ppo-reward-mode').value === 'game_placement'; }
  function rewardFields(resetValues = false) {
    const placement = placementMode();
    $('ppo-placement-weight').disabled = !placement;
    $('ppo-gae').disabled = placement;
    $('ppo-eval-hands').disabled = placement;
    $('ppo-milestone-hands').disabled = placement || !$('ppo-milestones-enabled').checked;
    for (const id of ['ppo-target', 'ppo-rollout', 'ppo-milestone-interval']) {
      $(id).min = placement ? '100' : '1';
      $(id).step = placement ? '100' : '1';
    }
    if (resetValues) {
      $('ppo-gae').value = placement ? '1' : '0.95';
      $('ppo-eval-hands').value = placement ? '100' : '10';
      if (placement) $('ppo-milestone-hands').value = '100';
    }
    const weight = Number($('ppo-placement-weight').value);
    message('ppo-reward-note', placement
      ? `${number(weight)}% game placement + ${number(100 - weight)}% bounded chip profit. Full 100-hand games include public scores and hand progress. Checkpoints are selected using duplicate-round placement and chip score. Start a new job to change the reward.`
      : 'Legacy reward: net chip profit for each hand. Existing checkpoints retain this objective when resumed.');
  }
  function milestoneSettings() {
    const enabled = $('ppo-milestones-enabled').checked;
    return {
      enabled,
      interval: enabled ? inputNumber('ppo-milestone-interval', 'Milestone interval', 1, 10000000, true) : 50000,
      eval_hands: enabled ? inputNumber('ppo-milestone-hands', 'Milestone evaluation hands', 1, 1000, true) : 100,
      eval_seeds: enabled ? inputNumber('ppo-milestone-seeds', 'Milestone seed groups', 1, 20, true) : 3
    };
  }
  function payload() {
    const checkpoint = sourceCheckpoint(), sizes = tableSizes();
    const mode = $('ppo-training-mode').value;
    if (!['bots', 'self_play', 'mixed'].includes(mode)) throw new Error('Choose a valid training mode.');
    if (mode !== 'bots' && state.self_play_available !== true) throw new Error('Self-play requires the updated dashboard server. Restart it to load this feature.');
    if (!checkpoint || !checkpoint.eligible || checkpoint.status !== 'completed') throw new Error('Choose an eligible, completed Behaviour Cloning checkpoint.');
    if (!sizes.length) throw new Error('Choose at least one table size.');
    const milestones = milestoneSettings();
    const minimum = milestones.enabled ? 5 : Math.max(...sizes) - 1;
    if (opponents.size < minimum || opponents.size > 60) throw new Error(milestones.enabled
      ? 'Choose 5–60 distinct benchmark opponents. Milestone evaluations always include 6-player tables, including in self-play mode.'
      : `Choose ${minimum}–60 distinct benchmark opponents for these validation table sizes, including in self-play mode.`);
    if (state.development_mode && $('ppo-device').value === 'mps') throw new Error('This test workspace uses CPU to leave the main training GPU available.');
    const settings = {
      name: $('ppo-name').value.trim(), source_training_id: String(checkpoint.id), opponents: [...opponents], table_sizes: sizes,
      target_episodes: inputNumber('ppo-target', 'Training episodes', 1, 10000000, true),
      rollout_episodes: inputNumber('ppo-rollout', 'Episodes per rollout', 1, 10000, true),
      update_epochs: inputNumber('ppo-update-epochs', 'Update passes', 1, 20, true),
      batch_size: inputNumber('ppo-batch', 'Batch size', 16, 8192, true),
      learning_rate: inputNumber('ppo-learning-rate', 'Policy learning rate', 0.000001, 0.01),
      critic_learning_rate: inputNumber('ppo-critic-rate', 'Value learning rate', 0.000001, 0.01),
      clip_ratio: inputNumber('ppo-clip', 'Clip ratio', 0.01, 0.5),
      entropy_coef: inputNumber('ppo-entropy', 'Entropy weight', 0, 1),
      bc_coef: inputNumber('ppo-bc-weight', 'BC anchor weight', 0, 10),
      target_kl: inputNumber('ppo-kl', 'Target KL', 0.001, 0.5),
      gae_lambda: inputNumber('ppo-gae', 'GAE lambda', 0, 1),
      eval_interval: inputNumber('ppo-eval-interval', 'Evaluation interval', 1, 1000, true),
      eval_hands: inputNumber('ppo-eval-hands', 'Evaluation hands', 1, 100, true),
      eval_seeds: inputNumber('ppo-eval-seeds', 'Evaluation seeds', 1, 20, true),
      seed: inputNumber('ppo-seed', 'Random seed', 0, 2147483647, true), device: $('ppo-device').value,
      milestone_settings: milestones
    };
    if (state.self_play_available === true) settings.training_mode = mode;
    if (mode === 'mixed') {
      settings.self_play_history_limit = inputNumber('ppo-history-limit', 'Recent historical versions', 1, 20, true);
      settings.self_play_history_interval = inputNumber('ppo-history-interval', 'Historical version interval', 1, 1000, true);
    }
    if (placementMode()) {
      if (state.placement_rewards_available !== true) throw new Error('Placement rewards require the updated preview server. Restart the preview to load this feature.');
      settings.reward_mode = 'game_placement';
      settings.placement_weight = inputNumber('ppo-placement-weight', 'Placement weight', 50, 100) / 100;
      settings.chip_scale_bb100 = 100;
      if (settings.target_episodes % 100 || settings.rollout_episodes % 100) throw new Error('Placement training needs episode and rollout counts divisible by 100, so every game finishes.');
      if (milestones.enabled && (milestones.interval % 100 || milestones.eval_hands !== 100)) throw new Error('Placement milestones need intervals divisible by 100 and 100-hand games.');
    } else settings.reward_mode = 'chips';
    return settings;
  }

  // Keep form choices stable while checkpoint and training progress refresh.
  function renderSources() {
    const checkpoints = list(state.checkpoints), signature = JSON.stringify(checkpoints);
    if (signature !== checkpointSignature) {
      checkpointSignature = signature;
      const selected = $('ppo-source').value;
      const options = [new Option('Choose a completed BC checkpoint', '')];
      for (const checkpoint of checkpoints) {
        const eligible = checkpoint.eligible && checkpoint.status === 'completed';
        const reason = eligible ? `best epoch ${number(checkpoint.epoch)} · completed` : checkpoint.reason || `${checkpoint.status || 'unavailable'} · finish BC training first`;
        const option = new Option(`${checkpoint.name || checkpoint.id} · ${reason}`, String(checkpoint.id));
        option.disabled = !eligible; options.push(option);
      }
      if (!checkpoints.length) options[0] = new Option('No completed BC checkpoints yet', '');
      $('ppo-source').replaceChildren(...options);
      if (checkpoints.some(checkpoint => String(checkpoint.id) === selected)) $('ppo-source').value = selected;
    }
    updateSourceDetail();
  }
  function updateSourceDetail() {
    const checkpoint = sourceCheckpoint();
    if (!checkpoint) { message('ppo-source-detail', 'Choose a completed BC checkpoint. Running BC jobs become available after they finish.'); return; }
    if (!checkpoint.eligible || checkpoint.status !== 'completed') {
      message('ppo-source-detail', checkpoint.reason || 'This BC checkpoint is not ready for PPO training.', true); return;
    }
    message('ppo-source-detail', `${checkpoint.name || checkpoint.id} · ${number(checkpoint.records)} training decisions · ${checkpoint.device || 'device recorded with source'}. PPO starts from its best validation snapshot and saves a separate agent.`);
  }
  function renderOpponents() {
    const agents = list(state.agents), available = new Set(agents.map(agent => String(agent.id)));
    for (const id of opponents) if (!available.has(id)) opponents.delete(id);
    const signature = JSON.stringify(agents.map(agent => [agent.id, agent.name, agent.hash]));
    if (signature === opponentSignature) return;
    opponentSignature = signature;
    $('ppo-opponent-list').replaceChildren();
    if (!agents.length) empty($('ppo-opponent-list'), 'Register opponents first', 'Your registered arena bots will appear here.');
    for (const agent of agents) {
      const row = element('label', null, 'lab-check-row'), input = element('input'), info = element('span', null, 'agent-info');
      input.type = 'checkbox'; input.value = String(agent.id); input.checked = opponents.has(input.value);
      input.addEventListener('change', () => { input.checked ? opponents.add(input.value) : opponents.delete(input.value); updateEstimate(); });
      info.append(element('strong', agent.name || agent.id), element('small', `Version ${String(agent.hash || agent.id).slice(0, 12)}`));
      row.append(input, info); $('ppo-opponent-list').append(row);
    }
  }
  function updateEstimate() {
    trainingFields();
    const enabled = $('ppo-milestones-enabled').checked;
    $('ppo-milestone-inputs').querySelectorAll('input').forEach(input => { input.disabled = !enabled; });
    rewardFields();
    try {
      const settings = payload(), rollouts = Math.ceil(settings.target_episodes / settings.rollout_episodes);
      const milestones = settings.milestone_settings;
      const milestoneNote = milestones.enabled ? ` Milestones every ${number(milestones.interval)} episodes use ${number(milestones.eval_hands)} hands per game and ${number(milestones.eval_seeds)} seed groups across 4/5/6 players.` : ' Milestone evaluations are off.';
      const games = placementMode() ? ` · ${number(settings.target_episodes / 100)} complete training games` : '';
      message('ppo-estimate', `${trainingModeLabel(settings.training_mode || 'bots')} · ${number(settings.target_episodes)} training episodes${games} · about ${number(rollouts)} rollouts · ${settings.table_sizes.join(', ')} players · ${number(settings.opponents.length)} fixed benchmark opponents. Validation runs every ${number(settings.eval_interval)} rollouts; a separate held-out test evaluates training best at the end.${milestoneNote}`);
    } catch (error) { message('ppo-estimate', error.message, true); }
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
    return job.config?.reward_mode === 'game_placement' || job.selection_metric === 'placement_chip_score' || job.snapshot?.best?.selection_metric === 'placement_chip_score';
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
    button.dataset.ppoId = String(id); button.dataset.ppoAction = action; button.dataset.ppoSource = source;
    button.disabled = pending.has(jobKey(id, source));
    return button;
  }
  function snapshotButtons(job, source = 'local', milestone = null) {
    const tools = element('div', null, 'run-tools');
    const label = milestone ? 'milestone' : 'training-best';
    for (const [format, text] of [['zip', 'agent ZIP'], ['checkpoint', 'checkpoint']]) {
      const button = actionButton(job.id, 'snapshot', `Download ${label} ${text}`, source);
      button.dataset.ppoKind = milestone ? 'milestone' : 'best';
      button.dataset.ppoFormat = format;
      if (milestone) button.dataset.ppoMilestone = milestone.id;
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
    const status = element('div', null, 'ppo-action-status');
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
      details.dataset.ppoDetails = `milestone:${snapshot.id}`;
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
      if (snapshot.resumable === false) details.append(element('p', 'This is a BC checkpoint. It supplies the starting policy rather than a resumable PPO iteration.', 'small muted'));
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
    svg.setAttribute('aria-label', placement ? 'PPO game-placement reward and duplicate-round validation score, compared with the BC baseline.' : 'Poker results during PPO: validation and training big blinds per 100 hands, compared with the initial BC validation baseline.');
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
    bar.max = Math.max(1, target); bar.value = Math.max(0, Math.min(target, episodes)); bar.setAttribute('aria-label', 'Completed PPO training episodes');
    progress.append(bar, element('span', `${number(episodes)} / ${number(target)} training episodes · ${Math.min(100, episodes / target * 100).toFixed(1)}%`, 'small muted')); node.append(progress);
    const facts = element('div', null, 'lab-facts');
    const trainingMode = config.training_mode || job.training_mode || 'bots';
    facts.append(element('span', `Mode: ${trainingModeLabel(trainingMode)}`), element('span', `Iteration ${number(job.iteration || 0)}`), element('span', `${number(job.decisions || 0)} learner decisions`));
    if (job.device || config.device) facts.append(element('span', `Device: ${job.device || config.device}`));
    if (finite(job.elapsed_seconds)) facts.append(element('span', `Elapsed ${duration(job.elapsed_seconds)}`));
    if (placement && finite(latest.train_reward)) facts.append(element('span', `Latest game reward: ${number(latest.train_reward)}`));
    if (finite(latest.train_bb100)) facts.append(element('span', `${placement ? 'Latest chip profit' : 'Latest training return'}: ${number(latest.train_bb100)} bb/100`));
    node.append(facts);
    const rewardLabel = placement ? `${number(Number(config.placement_weight ?? 0.9) * 100)}% game placement / ${number((1 - Number(config.placement_weight ?? 0.9)) * 100)}% bounded chips · 100-hand games · public game context` : 'Net chip profit per hand · legacy reward';
    node.append(element('p', `Reward: ${rewardLabel}.`, 'notice'));
    if (trainingMode !== 'bots') {
      node.append(element('p', 'Self-play returns measure performance against changing opponents. Chip profit near zero does not establish whether the policy is improving. Fixed external validation selects the best saved policy.', 'notice'));
      const selfPlay = job.self_play || latest.self_play;
      if (selfPlay) {
        const counts = selfPlay.seat_counts || {};
        node.append(element('p', `Latest rollout: ${number(selfPlay.games)} training games · current-policy seats ${number(counts.current)} · historical seats ${number(counts.historical)} · registered-bot seats ${number(counts.bots)}. ${number(selfPlay.history_versions_available)} past versions available for the next rollout.`, 'small muted'));
      }
    }
    const text = [job.phase, progressText(job)].filter(Boolean).join(' · ');
    if (text) node.append(element('p', text, 'small muted'));
    if (job.error) node.append(element('p', job.error, 'notice error'));
    const tools = element('div', null, 'run-tools');
    if (['queued', 'running'].includes(job.status) && !job.pause_requested) tools.append(actionButton(job.id, 'pause', 'Pause PPO'));
    if (['paused', 'interrupted', 'failed'].includes(job.status)) tools.append(actionButton(job.id, 'resume', 'Resume PPO'));
    if (job.status === 'completed' && !job.agent_id) tools.append(actionButton(job.id, 'export', 'Create PPO agent'));
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
        button.dataset.ppoFolder = folder; tools.append(button);
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
      node.append(element('p', Number(job.best_iteration) === 0 ? 'Training best: initial BC checkpoint (iteration 0).' : `Training best: PPO iteration ${number(job.best_iteration)}.`, 'small ppo-best-policy'));
    }
    const chart = evaluationPlot(job); if (chart) node.append(chart);
    node.append(element('p', 'Training best may be the original BC policy if PPO does not improve validation. PPO loss measures the update objective, not playing strength.', 'small muted'));
    node.append(trainingSnapshot(job), renderMilestones(job));
    const details = element('details', null, 'raw'); details.append(element('summary', 'Results by table size, update diagnostics and settings'));
    details.dataset.ppoDetails = 'training';
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
      details.append(table(history.map(row => [row.iteration, number(row.episodes), number(row.train_bb100), number(row.validation_bb100), precise(row.policy_loss), precise(row.value_loss), precise(row.entropy), precise(row.approx_kl), finite(row.clip_fraction) ? `${number(Number(row.clip_fraction) * 100)}%` : '—', precise(row.bc_kl), number(row.updates), row.early_stopped == null ? '—' : row.early_stopped ? 'Yes' : 'No']), ['Iteration', 'Episodes', 'Train bb/100', 'Validation bb/100', 'Policy loss', 'Value loss', 'Entropy', 'Approx. KL', 'Clipped', 'BC KL', 'Updates', 'KL early stop']));
      if (placement) details.append(table(history.map(row => [row.iteration, number(row.episodes), number(row.train_reward), number(row.validation_score)]), ['Iteration', 'Episodes', 'Training · game reward', 'Validation · round score']));
      const selfPlayHistory = history.filter(row => row.self_play);
      if (selfPlayHistory.length) {
        details.append(element('h3', 'Self-play opponents', 'lab-subheading'));
        details.append(element('p', 'Seat counts are summed across games in each rollout. The current-policy iteration identifies the frozen copy used for those games. Historical versions available describe the pool for the next rollout.', 'small muted'));
        details.append(table(selfPlayHistory.map(row => {
          const selfPlay = row.self_play, counts = selfPlay.seat_counts || {};
          return [row.iteration, number(selfPlay.current_iteration), number(selfPlay.games), number(counts.current), number(counts.historical), number(counts.bots), list(selfPlay.history_iterations_used).join(', ') || 'None', list(selfPlay.history_iterations_available).join(', ') || 'None'];
        }), ['Update iteration', 'Current-policy iteration', 'Games', 'Current seats', 'Historical seats', 'Registered seats', 'Historical iterations used', 'Available next rollout']));
      }
    }
    details.append(element('h3', 'Evaluation details', 'lab-subheading'), element('pre', JSON.stringify({ baseline: job.baseline || null, latest_validation: job.latest_validation || job.validation || null, best_validation: savedValidation, best_score: job.best_score ?? null, baseline_test: job.baseline_test || null, test: job.test || null }, null, 2)));
    details.append(element('h3', 'Training settings', 'lab-subheading'), element('pre', JSON.stringify(config, null, 2)));
    node.append(details);
    node.append(actionStatus(job));
  }
  function renderJobs() {
    const jobs = list(state.ppo_jobs), ids = new Set(jobs.map(job => String(job.id))), parent = $('ppo-jobs');
    $('ppo-job-count').textContent = `${jobs.length} PPO job${jobs.length === 1 ? '' : 's'}`;
    for (const [id, card] of jobCards) if (!ids.has(id)) { card.node.remove(); jobCards.delete(id); }
    if (!jobs.length) { empty(parent, 'Continue learning from your BC student', 'Choose a completed checkpoint and opponents, then start a PPO experiment.'); return; }
    if (parent.querySelector(':scope > .empty')) parent.replaceChildren();
    for (const job of jobs) {
      const id = String(job.id), signature = JSON.stringify([job, state.folder_actions_available, state.finder_available, pending.has(id), actionMessages.get(id)]);
      let card = jobCards.get(id);
      if (!card) { card = { node: element('article', null, 'lab-card ppo-job'), signature: '' }; jobCards.set(id, card); parent.append(card.node); }
      if (card.signature === signature) continue;
      const opened = new Set(Array.from(card.node.querySelectorAll('details[open][data-ppo-details]'), details => details.dataset.ppoDetails));
      renderJob(card.node, job);
      for (const details of card.node.querySelectorAll('details[data-ppo-details]')) details.open = opened.has(details.dataset.ppoDetails);
      card.signature = signature;
    }
    jobs.forEach((job, index) => {
      const node = jobCards.get(String(job.id)).node;
      if (parent.children[index] !== node) parent.insertBefore(node, parent.children[index] || null);
    });
  }
  function renderLiveJobs() {
    const wasHidden = $('ppo-live-section').hidden;
    $('ppo-live-section').hidden = state.live_source_enabled !== true;
    $('ppo-history-shortcut').hidden = state.live_source_enabled !== true;
    if (state.live_source_enabled !== true) return;
    const jobs = list(state.live_jobs), ids = new Set(jobs.map(job => String(job.id))), parent = $('ppo-live-jobs');
    $('ppo-live-count').textContent = `${number(jobs.length)} source job${jobs.length === 1 ? '' : 's'}`;
    for (const [id, card] of liveCards) if (!ids.has(id)) { card.node.remove(); liveCards.delete(id); }
    if (!jobs.length) {
      empty(parent, 'No main-dashboard jobs available', state.live_source_error || 'The read-only source has no PPO jobs to display.'); return;
    }
    if (parent.querySelector(':scope > .empty')) parent.replaceChildren();
    for (const job of jobs) {
      const id = String(job.id), key = jobKey(id, 'live');
      const signature = JSON.stringify([job, pending.has(key), actionMessages.get(key)]);
      let card = liveCards.get(id);
      if (!card) { card = { node: element('article', null, 'lab-card ppo-job ppo-live-job'), signature: '' }; liveCards.set(id, card); parent.append(card.node); }
      if (card.signature === signature) continue;
      const heading = element('div', null, 'lab-card-heading'), title = element('div', null, 'run-title');
      title.append(element('strong', job.name || job.id), element('small', `Created ${date(job.created)} · original dashboard · read-only`));
      const badge = element('span', job.status || 'unknown', 'tag');
      if (/^(queued|running|paused|interrupted|failed|completed)$/.test(job.status || '')) badge.classList.add(`status-${job.status}`);
      heading.append(title, badge);
      const progress = `${number(job.episodes)} / ${number(job.target_episodes)} training episodes · iteration ${number(job.iteration)}`;
      const downloadsOpen = card.node.querySelector('details')?.open || false;
      card.node.replaceChildren(heading, element('p', progress, 'small muted'));
      const config = job.config || {}, placement = placementJob(job);
      const mode = trainingModeLabel(config.training_mode || job.training_mode || 'bots');
      const reward = placement
        ? `${number(Number(config.placement_weight ?? 0.9) * 100)}% game placement + ${number((1 - Number(config.placement_weight ?? 0.9)) * 100)}% bounded chip profit`
        : 'Net chip profit per hand';
      card.node.append(element('p', `${mode} · Reward: ${reward}.`, 'small muted'));
      const scores = element('div', null, 'ppo-score-grid');
      const best = job.best_validation || (job.status === 'completed' ? job.validation : null);
      scores.append(
        scoreCard('Initial BC · validation', baselineEvaluation(job), 'Starting policy on fixed validation games', placement),
        scoreCard('Training best · validation', best, 'Best saved validation result; not the latest training reward', placement),
        scoreCard('Initial BC · held-out test', job.baseline_test, 'Starting policy on separate test games', placement),
        scoreCard('Training best · held-out test', job.test, 'Selected policy on the same test games', placement)
      );
      const chart = evaluationPlot(job);
      card.node.append(chart || element('p', 'No training reward measurements were saved for this run.', 'small muted'));
      card.node.append(scores);
      // Keep source jobs read-only: never add local resume, pause or export actions.
      const downloads = element('details', null, 'raw');
      downloads.append(element('summary', 'Saved checkpoint downloads'), trainingSnapshot(job, 'live'));
      downloads.open = downloadsOpen;
      card.node.append(downloads, actionStatus(job, 'live'));
      card.signature = signature;
    }
    jobs.forEach((job, index) => {
      const node = liveCards.get(String(job.id)).node;
      if (parent.children[index] !== node) parent.insertBefore(node, parent.children[index] || null);
    });
    if (wasHidden && location.hash === '#ppo-live-section') {
      requestAnimationFrame(() => $('ppo-live-section').scrollIntoView({ block: 'start' }));
    }
  }
  async function refresh() {
    if (refreshing) return;
    refreshing = true;
    try {
      state = await api('/api/ppo');
      window.dispatchEvent(new CustomEvent('arena-state', { detail: { source: 'ppo', state } }));
      const development = state.development_mode === true;
      $('ppo-development-notice').hidden = !development;
      $('ppo-device-note').hidden = !development;
      $('ppo-device').querySelector('option[value="mps"]').disabled = development;
      $('ppo-device').querySelector('option[value="cuda"]').disabled = development;
      $('ppo-device').querySelector('option[value="auto"]').textContent = development ? 'Automatic · CPU in this test workspace' : 'Automatic · detect available GPU';
      if (development) $('ppo-device').value = 'cpu';
      renderSources(); renderOpponents(); updateEstimate(); renderJobs(); renderLiveJobs(); $('ppo-error').hidden = true;
    } catch (error) {
      $('ppo-error').textContent = `PPO status unavailable: ${error.message} Retrying automatically.`; $('ppo-error').hidden = false;
    } finally { refreshing = false; }
  }

  $('ppo-source').addEventListener('change', () => { updateSourceDetail(); updateEstimate(); });
  $('ppo-reward-mode').addEventListener('change', () => { rewardFields(true); updateEstimate(); });
  $('ppo-opponents-all').onclick = () => {
    for (const agent of list(state.agents)) opponents.add(String(agent.id));
    $('ppo-opponent-list').querySelectorAll('input').forEach(input => { input.checked = true; }); updateEstimate();
  };
  $('ppo-opponents-none').onclick = () => {
    opponents.clear(); $('ppo-opponent-list').querySelectorAll('input').forEach(input => { input.checked = false; }); updateEstimate();
  };
  $('ppo-form').addEventListener('input', updateEstimate);
  $('ppo-form').addEventListener('change', updateEstimate);
  $('ppo-form').addEventListener('submit', async event => {
    event.preventDefault(); $('ppo-create').disabled = true;
    try {
      const settings = payload();
      message('ppo-status', 'Saving PPO experiment…');
      await api('/api/ppo', settings);
      message('ppo-status', 'PPO training queued. Follow its progress below; the BC source checkpoint is preserved.'); await refresh();
    } catch (error) { message('ppo-status', error.message, true); }
    finally { $('ppo-create').disabled = false; }
  });
  async function handleJobAction(event) {
    const button = event.target.closest('button[data-ppo-action]');
    if (!button || button.disabled) return;
    const id = button.dataset.ppoId, action = button.dataset.ppoAction, folder = button.dataset.ppoFolder;
    const source = button.dataset.ppoSource || 'local', key = jobKey(id, source);
    if (source === 'live' && action !== 'snapshot') return;
    if (pending.has(key)) return;
    pending.add(key); button.disabled = true;
    function actionMessage(text, error = false, snapshot = null) {
      actionMessages.set(key, { text, error, snapshot }); message('ppo-status', text, error);
      const status = button.closest('.ppo-job')?.querySelector('.ppo-action-status');
      if (status) { status.textContent = text; status.hidden = false; status.classList.toggle('error-text', error); }
    }
    try {
      if (action === 'snapshot') {
        const kind = source === 'live' ? 'best' : button.dataset.ppoKind;
        const body = { kind };
        if (kind === 'milestone') body.milestone_id = button.dataset.ppoMilestone;
        actionMessage(source === 'live' ? 'Copying the saved main-dashboard checkpoint and checking its agent in this preview…' : 'Copying the saved checkpoint and checking its downloadable agent…');
        const prefix = source === 'live' ? '/api/ppo/live/' : '/api/ppo/';
        const result = await api(`${prefix}${encodeURIComponent(id)}/snapshot`, body);
        const url = button.dataset.ppoFormat === 'checkpoint' ? result.checkpoint_url : result.zip_url;
        const link = downloadLink(url, 'Download saved snapshot');
        if (!link) throw new Error('The snapshot was prepared, but the server did not return a valid download link.');
        actionMessage('Snapshot ready. Your download is starting; both saved files are available below.', false, result);
        link.hidden = true; document.body.append(link); link.click(); link.remove();
      } else if (action === 'open-folder') {
        if (state.folder_actions_available !== true || state.finder_available !== true) throw new Error('Finder controls are unavailable on this dashboard server.');
        actionMessage(`Opening ${folder} folder in Finder…`);
        const result = await api(`/api/ppo/${encodeURIComponent(id)}/open-folder`, { folder });
        actionMessage(`Opened ${folder} folder in Finder: ${result.path}`);
      } else {
        actionMessage(action === 'export' ? 'Packaging and checking the selected policy…' : 'Saving request…');
        await api(`/api/ppo/${encodeURIComponent(id)}/${action}`, {});
        const messages = { pause: 'Pause requested. Completed iterations are preserved; an unfinished iteration is replayed on resume.', resume: 'PPO queued to resume from its saved checkpoint.', export: 'The selected PPO policy is registered as a separate agent. Test it in the arena with fresh seeds.' };
        actionMessage(messages[action] || 'Saved.');
      }
    } catch (error) { actionMessage(error.message, true); }
    finally { pending.delete(key); button.disabled = false; await refresh(); renderJobs(); renderLiveJobs(); }
  }
  $('ppo-jobs').addEventListener('click', handleJobAction);
  $('ppo-live-jobs').addEventListener('click', handleJobAction);
  refresh(); setInterval(refresh, 2000);
})();
