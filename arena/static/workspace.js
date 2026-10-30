/* View navigation and the published research catalogue. No build step required. */
(() => {
  'use strict';
  const element = id => document.getElementById(id);
  const rows = value => Array.isArray(value) ? value : [];
  const format = value => Number(value || 0).toLocaleString();
  const views = {
    overview: ['Overview', 'FROM IDEA TO EVIDENCE', 'Build a bot. Test what works.', 'A local workspace for reproducible poker experiments.'],
    models: ['Model library', 'START WITH YOUR BASELINES', 'Every model has a story.', 'Register agents, select a comparison pool and trace original sources.'],
    data: ['Data', 'COLLECT DEMONSTRATIONS', 'Good experiments start with data.', 'Record teacher decisions or import poker hand histories.'],
    training: ['Training', 'CHOOSE A LEARNING PATH', 'From imitation to improvement.', 'Train a student, then compare reinforcement learning approaches.'],
    evaluate: ['Evaluate', 'MEASURE PLAYING STRENGTH', 'Put your hypothesis to the test.', 'Configure comparable runs and inspect results for every table size.'],
    experiments: ['Experiments', 'KEEP THE EVIDENCE', 'What worked. What comes next.', 'Explore the research snapshot and the history of your own runs.'],
  };
  let trainingMethod = 'cloning';

  function node(tag, text, className) {
    const result = document.createElement(tag);
    if (text != null) result.textContent = String(text);
    if (className) result.className = className;
    return result;
  }

  function navigate(moveFocus = false) {
    const requested = location.hash.slice(1) || 'overview';
    const target = element(requested);
    const view = target?.closest('.workspace-view')?.id || 'overview';
    const [label, eyebrow, title, description] = views[view];
    document.querySelectorAll('.workspace-view').forEach(section => { section.hidden = section.id !== view; });
    element('view-breadcrumb').textContent = label;
    element('view-eyebrow').textContent = eyebrow;
    element('view-title').textContent = title;
    element('view-description').textContent = description;
    document.title = `${label} · Poker Bot Challenge Competition`;
    document.querySelectorAll('.main-navigation a').forEach(link => {
      const active = link.hash === `#${view}`;
      link.classList.toggle('active', active);
      if (active) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
    if (view === 'training') {
      const method = target?.closest('#cloning, #ppo, #grpo')?.id;
      if (method) trainingMethod = method;
    }
    ['cloning', 'ppo', 'grpo'].forEach(id => { element(id).hidden = id !== trainingMethod; });
    document.querySelectorAll('.method-navigation a').forEach(link => {
      const active = link.hash === `#${trainingMethod}`;
      link.classList.toggle('active', active);
      if (active) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
    if (moveFocus) {
      // Native hash links still work when their destination lives in another view.
      requestAnimationFrame(() => {
        const destination = target && target.id !== view ? target : element('view-title');
        if (!destination.hasAttribute('tabindex')) destination.setAttribute('tabindex', '-1');
        destination.focus({ preventScroll: true });
        destination.scrollIntoView({ block: 'start', behavior: 'instant' });
      });
    }
  }
  window.addEventListener('hashchange', () => navigate(true));
  document.addEventListener('click', event => {
    const link = event.target.closest('a[href^="#"]');
    if (link?.classList.contains('skip-link')) {
      event.preventDefault();
      element('workspace').focus();
      return;
    }
    if (link && link.hash === location.hash) navigate(true);
  });
  navigate(Boolean(location.hash));

  function sourceLink(url, label) {
    try {
      const destination = new URL(url);
      if (!['https:', 'http:'].includes(destination.protocol)) return node('span', 'Source unverified', 'muted');
      const link = node('a', label, 'source-link');
      link.href = destination.href;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      return link;
    } catch {
      return node('span', 'Source unverified', 'muted');
    }
  }

  function renderResearch(research) {
    const latest = research.latest_experiment;
    const highlight = element('research-current');
    highlight.replaceChildren();
    if (latest) {
      const heading = node('h2', latest.name);
      heading.id = 'research-title';
      highlight.append(heading, node('p', `${latest.starting_model || 'Saved policy'} → ${latest.name}`, 'research-lineage'));
      const status = node('span', `${latest.status || 'Recorded'} in saved workspace`, 'tag status-paused');
      highlight.append(status);
      if (latest.episodes_target) {
        const progress = node('progress');
        progress.max = Number(latest.episodes_target);
        progress.value = Number(latest.episodes_completed || 0);
        progress.setAttribute('aria-label', 'Saved experiment training progress');
        const box = node('div', null, 'lab-progress research-progress');
        box.append(node('strong', `${format(latest.episodes_completed)} / ${format(latest.episodes_target)} episodes`), progress);
        highlight.append(box);
      }
      highlight.append(node('p', latest.reward, 'small muted'));
      highlight.append(node('p', latest.note || 'Historical saved state; not a running job on this installation.', 'snapshot-note'));
    } else {
      const heading = node('h2', 'No published experiment yet'); heading.id = 'research-title';
      highlight.append(heading, node('p', 'Your local experiments appear in the training and evaluation views.', 'muted'));
    }
    element('research-as-of').textContent = research.as_of ? `SNAPSHOT · ${research.as_of}` : 'SAVED SNAPSHOT';
    element('research-scope').textContent = research.scope || 'Reference history; local jobs are tracked separately.';
    const methods = element('research-methods');
    methods.replaceChildren();
    rows(research.methods).forEach(method => {
      const card = node('article', null, 'method-card');
      card.append(node('h3', method.title), node('span', method.status, 'method-status'), node('p', method.summary));
      const details = node('details', null, 'raw');
      details.append(node('summary', 'Evidence & limitations'), node('p', method.evidence), node('p', method.limitations, 'muted'));
      card.append(details); methods.append(card);
    });
    const history = element('research-history');
    history.replaceChildren();
    rows(research.history).forEach(experiment => {
      const entry = node('article', null, 'history-entry');
      const header = node('div', null, 'teacher-heading');
      header.append(node('h3', experiment.name), node('span', experiment.status, 'tag'));
      entry.append(header);
      const facts = [experiment.date, experiment.starting_model ? `From ${experiment.starting_model}` : '', rows(experiment.table_sizes).length ? `${experiment.table_sizes.join(' / ')} players` : ''].filter(Boolean);
      entry.append(node('p', facts.join(' · '), 'small muted'));
      if (experiment.episodes_target) entry.append(node('p', `${format(experiment.episodes_completed)} / ${format(experiment.episodes_target)} episodes`, 'small'));
      entry.append(node('p', experiment.note));
      history.append(entry);
    });
    if (!history.children.length) history.append(node('p', 'No published history is available.', 'muted'));
  }

  function renderSources(sources) {
    const parent = element('model-sources');
    parent.replaceChildren();
    if (!rows(sources.models).length) {
      parent.append(node('p', 'No source catalogue is available. Record each model’s author, original URL and licence when adding it.', 'muted'));
      return;
    }
    const wrap = node('div', null, 'table-wrap');
    const table = node('table', null, 'source-table');
    const header = node('thead'), headerRow = node('tr');
    ['Model / version', 'Original author', 'Licence', 'Source'].forEach(label => {
      const cell = node('th', label); cell.scope = 'col'; headerRow.append(cell);
    });
    header.append(headerRow); table.append(header);
    const body = node('tbody');
    rows(sources.models).forEach(model => {
      const row = node('tr'), title = node('td');
      title.append(node('strong', model.name));
      if (model.note) {
        const note = node('details', null, 'raw');
        note.append(node('summary', 'Provenance notes'), node('p', model.note)); title.append(note);
      }
      const linkCell = node('td'); linkCell.append(sourceLink(model.source_url, 'Original source ↗'));
      row.append(title, node('td', model.author || 'Author unverified'), node('td', model.license || 'Not verified'), linkCell);
      body.append(row);
    });
    table.append(body); wrap.append(table); parent.append(wrap);
    const credits = node('a', 'Read full credits, dataset sources & licence notes →', 'text-button');
    credits.href = '/docs/CREDITS.md'; credits.target = '_blank'; credits.rel = 'noopener';
    parent.append(credits);
  }

  async function loadResearch() {
    try {
      const response = await fetch('/api/research', { cache: 'no-store' });
      if (!response.ok) throw new Error('Research catalogue unavailable');
      const result = await response.json();
      renderResearch(result.experiments || {});
      renderSources(result.sources || {});
    } catch {
      renderResearch({}); renderSources({});
      element('research-scope').textContent = 'Could not load the published catalogue. Your local jobs are unaffected. Refresh to retry.';
    }
  }
  loadResearch();

  const localStates = {};
  let activitySignature = '';
  window.addEventListener('arena-state', event => {
    localStates[event.detail.source] = event.detail.state;
    const jobs = [
      ...rows(localStates.arena?.runs).map(job => ({ ...job, kind: 'Evaluation', href: '#queue' })),
      ...rows(localStates.lab?.datasets).map(job => ({ ...job, kind: 'Data', href: '#datasets' })),
      ...rows(localStates.lab?.training_jobs).map(job => ({ ...job, kind: 'BC', href: '#cloning' })),
      ...rows(localStates.ppo?.ppo_jobs).map(job => ({ ...job, kind: 'PPO', href: '#ppo' })),
      ...rows(localStates.grpo?.grpo_jobs).map(job => ({ ...job, kind: 'GRPO', href: '#grpo' })),
    ].filter(job => ['running', 'queued', 'pending', 'collecting', 'training', 'exporting'].includes(job.status));
    const signature = JSON.stringify([Object.keys(localStates).sort(), jobs.map(job => [job.id, job.name, job.status])]);
    if (signature === activitySignature) return;
    activitySignature = signature;
    const parent = element('workspace-activity'); parent.replaceChildren();
    if (!jobs.length) {
      const complete = ['arena', 'lab', 'ppo', 'grpo'].every(source => localStates[source]);
      const empty = node('div', null, 'empty');
      empty.append(node('strong', complete ? 'No active jobs on this installation' : 'Checking local job status…'));
      empty.append(node('span', complete ? 'Register an agent to start a benchmark, or choose a training method. Saved research history does not start jobs automatically.' : 'Waiting for all workspace services to respond.'));
      parent.append(empty); return;
    }
    jobs.forEach(job => {
      const link = node('a', null, 'activity-row'); link.href = job.href;
      const title = node('span'); title.append(node('strong', job.name || job.id), node('small', job.kind));
      link.append(title, node('span', job.status, 'tag status-running')); parent.append(link);
    });
  });
})();
