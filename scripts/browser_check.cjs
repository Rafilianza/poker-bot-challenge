/* An isolated browser smoke test. Install with npm ci; run with npm test. */
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { chromium } = require('playwright');

const root = path.resolve(__dirname, '..');
const windows = process.platform === 'win32';
const python = process.env.ARENA_PYTHON || path.join(root, '.venv', windows ? 'Scripts/python.exe' : 'bin/python');
const screenshots = path.join(root, 'test-results');

function startServer(dataDirectory) {
  // Port zero asks the OS for a free port. This never connects to a user's server.
  const server = spawn(python, ['run.py', '--port', '0', '--data-dir', dataDirectory], {
    cwd: root,
    env: { ...process.env, PYTHONUTF8: '1', PYTHONUNBUFFERED: '1' },
    detached: !windows,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let output = '';
  const ready = new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`Server startup timed out.\n${output}`)), 30000);
    server.on('error', error => {
      clearTimeout(timer);
      reject(new Error(`Cannot start Python. Create .venv or set ARENA_PYTHON. ${error.message}`));
    });
    server.on('exit', code => {
      clearTimeout(timer);
      reject(new Error(`Server exited with code ${code}.\n${output}`));
    });
    server.stderr.on('data', chunk => { output = (output + chunk).slice(-12000); });
    server.stdout.on('data', chunk => {
      output = (output + chunk).slice(-12000);
      const match = output.match(/Poker Arena: (http:\/\/127\.0\.0\.1:\d+)/);
      if (match) {
        clearTimeout(timer);
        resolve(match[1]);
      }
    });
  });
  return { server, ready };
}

async function stopServer(server) {
  if (!server.pid) return;
  if (windows) {
    // run.py has a server child; terminate this test's process tree together.
    await new Promise(resolve => {
      const cleanup = spawn('taskkill', ['/PID', String(server.pid), '/T', '/F'], { stdio: 'ignore' });
      cleanup.once('close', resolve);
      cleanup.once('error', resolve);
    });
  } else {
    try { process.kill(-server.pid, 'SIGTERM'); }
    catch (error) { if (error.code !== 'ESRCH') throw error; }
  }
  if (server.exitCode === null && server.signalCode === null) {
    await new Promise(resolve => {
      const timer = setTimeout(resolve, 5000);
      server.once('exit', () => { clearTimeout(timer); resolve(); });
    });
  }
}

async function goTo(page, hash, view) {
  await page.evaluate(destination => { location.hash = destination; }, hash);
  await page.locator(`#${view}`).waitFor({ state: 'visible' });
  assert.equal(await page.locator('.workspace-view:visible').count(), 1, `One view at ${hash}`);
  assert.equal(await page.locator('.main-navigation a[aria-current="page"]').getAttribute('href'), `#${view}`);
}

async function checkWidth(page, label) {
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
  assert.equal(overflow, false, `Page overflow at ${label}`);
}

async function main() {
  const dataDirectory = await fs.mkdtemp(path.join(os.tmpdir(), 'poker-arena-browser-'));
  await fs.mkdir(screenshots, { recursive: true });
  const { server, ready } = startServer(dataDirectory);
  let browser;
  try {
    const address = await ready;
    const options = { headless: true };
    if (process.env.PLAYWRIGHT_CHANNEL) options.channel = process.env.PLAYWRIGHT_CHANNEL;
    browser = await chromium.launch(options);
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
    await page.goto(address);
    await page.getByText('Live · local server', { exact: true }).waitFor();
    await page.getByText('No active jobs on this installation', { exact: true }).waitFor();

    const research = await (await page.request.get(`${address}/api/research`)).json();
    assert.equal(research.sources.models.length, 17);
    assert.equal(research.experiments.latest_experiment.name, 'SELF-PPO');
    assert.equal(research.experiments.latest_experiment.status, 'paused');
    await page.locator('#research-current').getByRole('heading', { name: 'SELF-PPO', exact: true }).waitFor();
    assert.match(await page.locator('#research-current').innerText(), /not a running job/i);
    await page.screenshot({ path: path.join(screenshots, 'overview-desktop.png'), fullPage: true });

    const views = ['overview', 'models', 'data', 'training', 'evaluate', 'experiments'];
    for (const view of views) {
      await page.locator(`.main-navigation a[href="#${view}"]`).click();
      await page.locator(`#${view}`).waitFor({ state: 'visible' });
      await checkWidth(page, `desktop ${view}`);
    }
    assert.equal(await page.locator('#research-methods .method-card').count(), research.experiments.methods.length);
    assert.equal(await page.locator('#research-history .history-entry').count(), research.experiments.history.length);
    await goTo(page, 'sources', 'models');
    assert.equal(await page.locator('#model-sources tbody tr').count(), 17);
    assert.match(await page.locator('#sources').innerText(), /not automatically installed or bundled/i);
    await page.screenshot({ path: path.join(screenshots, 'models-desktop.png'), fullPage: true });

    for (const method of ['cloning', 'ppo', 'grpo']) {
      await goTo(page, method, 'training');
      await page.locator(`#${method}`).waitFor({ state: 'visible' });
      assert.equal(await page.locator('#training > .lab-panel:visible').count(), 1);
      assert.equal(await page.locator('.method-navigation a[aria-current="page"]').getAttribute('href'), `#${method}`);
      await checkWidth(page, `desktop ${method}`);
    }
    await page.screenshot({ path: path.join(screenshots, 'training-desktop.png'), fullPage: true });

    // Use actual upload controls and the real engine, with only repository examples.
    await goTo(page, 'models', 'models');
    await page.locator('#agent-files').setInputFiles(['calling_bot.py', 'cautious_bot.py', 'random_bot.py'].map(name => path.join(root, 'examples', name)));
    await page.locator('#upload-button').click();
    await page.getByText('3/3 agents registered.', { exact: true }).waitFor();
    await page.waitForFunction(() => document.querySelectorAll('#agent-list input').length === 3);
    await page.locator('#select-all').click();
    await goTo(page, 'launch', 'evaluate');
    await page.locator('#seeds').fill('demo1');
    await page.locator('#deals-preset').selectOption('10');
    await page.locator('#queue-button').click();
    await page.getByText('Run queued. Follow its progress in Experiments → Evaluation queue & history.', { exact: true }).waitFor();
    const state = await (await page.request.get(`${address}/api/state`)).json();
    assert.equal(state.runs.length, 1, 'The isolated workspace contains one evaluation run');
    const runId = state.runs[0].id;
    const deadline = Date.now() + 180000;
    let run;
    // Poll the same saved run that supplies the assertions below, outside the page.
    // A successful queue request or a UI refresh does not mean the run has finished.
    while (Date.now() < deadline) {
      const response = await page.request.get(`${address}/api/runs/${runId}`, { timeout: Math.min(15000, deadline - Date.now()) });
      assert.equal(response.ok(), true, 'Run detail request succeeds');
      run = await response.json();
      if (['completed', 'failed', 'cancelled'].includes(run.status)) break;
      await new Promise(resolve => setTimeout(resolve, Math.max(0, Math.min(1000, deadline - Date.now()))));
    }
    assert.ok(run && ['completed', 'failed', 'cancelled'].includes(run.status), `Run did not finish within 180 seconds: ${JSON.stringify(run?.progress)}`);
    assert.equal(run.status, 'completed', JSON.stringify(run.error || run.progress));
    assert.equal(run.leaderboard.length, 3);
    const games = run.results.flatMap(result => result.games);
    assert.equal(games.length, 15, 'Three agents, each at all five benchmark seats');
    for (const game of games) assert.deepEqual(game.verdicts, Array(5).fill('OK'));
    for (const entry of run.leaderboard) {
      assert.equal(entry.hands, 50);
      assert.equal(entry.failures, 0);
    }
    await goTo(page, 'rankings', 'evaluate');
    await page.locator('#leaderboard tbody tr').first().waitFor();
    await page.screenshot({ path: path.join(screenshots, 'evaluation-desktop.png'), fullPage: true });

    // The exported archive should download from a fresh installation, too.
    await goTo(page, 'models', 'models');
    const downloadEvent = page.waitForEvent('download');
    await page.locator('#agent-list').getByRole('link', { name: 'Download ZIP' }).first().click();
    const download = await downloadEvent;
    assert.match(download.suggestedFilename(), /^agent-[a-f0-9]{32}\.zip$/);
    assert.equal(await download.failure(), null);

    await page.setViewportSize({ width: 390, height: 844 });
    for (const view of views) {
      await goTo(page, view, view);
      await checkWidth(page, `mobile ${view}`);
    }
    for (const method of ['cloning', 'ppo', 'grpo']) {
      await goTo(page, method, 'training');
      await checkWidth(page, `mobile ${method}`);
    }
    await goTo(page, 'overview', 'overview');
    await page.screenshot({ path: path.join(screenshots, 'overview-mobile.png'), fullPage: true });
    await goTo(page, 'queue', 'experiments');
    await page.locator('#run-list .run > summary').first().click();
    await page.getByText('Hand history explorer', { exact: true }).first().waitFor();
    await checkWidth(page, 'mobile expanded run history');
    await page.screenshot({ path: path.join(screenshots, 'history-mobile.png'), fullPage: true });
    assert.deepEqual(errors, [], 'No JavaScript errors or browser console errors');
    console.log(JSON.stringify({ views: 6, trainingMethods: 3, sources: 17, registeredAgents: 3, benchmarkGames: games.length, benchmarkHands: 150, verdicts: 'all OK', mobileOverflow: false, browserErrors: errors, screenshots: 'test-results/' }, null, 2));
  } finally {
    try {
      if (browser) await browser.close();
    } finally {
      await stopServer(server);
      await fs.rm(dataDirectory, { recursive: true, force: true });
    }
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
