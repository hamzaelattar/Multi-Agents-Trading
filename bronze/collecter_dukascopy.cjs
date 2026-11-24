// Daily UTC checkpoints keep memory bounded and make retries independent of CSV merges.
const {createHash} = require('node:crypto');
const fs = require('node:fs/promises');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const {parseArgs} = require('node:util');
const {getHistoricalRates, URL_ROOT} = require('dukascopy-node');
const libraryVersion = '1.51.1'; // pinned in package-lock.json
const dayMs = 86_400_000;
const header = 'timestamp_utc,ask,bid,ask_volume,bid_volume';

function day(value) {
    const timestamp = Date.parse(`${value}T00:00:00.000Z`);
    if (!Number.isFinite(timestamp) || new Date(timestamp).toISOString().slice(0, 10) !== value) {
        throw new Error(`Date UTC invalide : ${value}`);
    }
    return timestamp;
}

function literal(value) {
    return `'${value.replaceAll("'", "''")}'`;
}

function decimal(value) {
    if (!Number.isFinite(value)) throw new Error('Une valeur de tick est absente ou non finie.');
    const text = String(value);
    if (!text.includes('e')) return text;
    const [coefficient, exponent] = text.split('e');
    const sign = coefficient.startsWith('-') ? '-' : '';
    const [whole, fraction = ''] = coefficient.replace(/^-/, '').split('.');
    const digits = whole + fraction;
    const point = whole.length + Number(exponent);
    if (point <= 0) return `${sign}0.${'0'.repeat(-point)}${digits}`;
    if (point >= digits.length) return sign + digits + '0'.repeat(point - digits.length);
    return `${sign}${digits.slice(0, point)}.${digits.slice(point)}`;
}

function toCsv(rows, from, to) {
    const lines = [header];
    for (const tick of rows) {
        if (tick.length !== 5 || !Number.isInteger(tick[0]) || tick[0] < from || tick[0] >= to) {
            throw new Error('Tick ou horodatage hors du schéma et de la journée demandés.');
        }
        lines.push([new Date(tick[0]).toISOString(), ...tick.slice(1).map(decimal)].join(','));
    }
    return lines.join('\n') + '\n';
}

function duckdbJson(binary, sql, cwd) {
    const result = spawnSync(binary, ['-batch', '-bail', '-json'], {input: sql, encoding: 'utf8', cwd});
    if (result.error) throw result.error;
    if (result.status !== 0) throw new Error(result.stderr.trim());
    return JSON.parse(result.stdout);
}

function validate(binary, file, expected, cwd) {
    const scan = `read_csv(${literal(file.replaceAll('\\', '/'))}, header=true, auto_detect=false,
        delim=',', parallel=false, columns={'timestamp_utc':'VARCHAR','ask':'VARCHAR','bid':'VARCHAR',
        'ask_volume':'VARCHAR','bid_volume':'VARCHAR'})`;
    const numbers = ['ask', 'bid', 'ask_volume', 'bid_volume'];
    const typed = numbers.map(field => `try_cast(${field} AS DOUBLE) AS d_${field}`).join(', ');
    const validNumbers = numbers.map(field => `coalesce(regexp_full_match(${field}, '[+-]?[0-9]+([.][0-9]+)?') AND isfinite(d_${field}), false)`).join(' AND ');
    const projection = `WITH typed AS (
        SELECT *, try_cast(timestamp_utc AS TIMESTAMPTZ) AS t, ${typed} FROM ${scan}
    ), sequenced AS (SELECT *, lag(t) OVER () AS previous FROM typed)`;
    const [summary] = duckdbJson(binary, `SET TimeZone='UTC'; SET threads=1;
        ${projection}
        SELECT count(*) AS ticks, min(t) AS first_utc, max(t) AS last_utc,
            count(DISTINCT date_trunc('minute',t)) AS minutes_with_ticks,
            count(DISTINCT date_trunc('hour',t)) AS hours_with_ticks,
            count(*) FILTER (WHERE t < previous) AS chronological_reversals,
            count(*) FILTER (WHERE t = previous) AS adjacent_equal_timestamps,
            count(*) - count(DISTINCT (timestamp_utc,ask,bid,ask_volume,bid_volume)) AS exact_duplicate_rows,
            count(*) FILTER (WHERE timestamp_utc IS NULL OR ask IS NULL OR bid IS NULL
                OR ask_volume IS NULL OR bid_volume IS NULL) AS rows_with_nulls,
            count(*) FILTER (WHERE t IS NULL OR NOT coalesce(regexp_full_match(timestamp_utc,
                '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}[.][0-9]{3}Z'),false)) AS invalid_timestamps,
            count(*) FILTER (WHERE NOT (${validNumbers})) AS invalid_numbers,
            count(*) FILTER (WHERE d_ask <= 0 OR d_bid <= 0) AS nonpositive_prices,
            count(*) FILTER (WHERE d_ask < d_bid) AS negative_spreads,
            count(*) FILTER (WHERE d_ask = d_bid) AS zero_spreads,
            count(*) FILTER (WHERE d_ask_volume < 0 OR d_bid_volume < 0) AS negative_volumes,
            max(d_ask_volume) AS max_ask_volume, max(d_bid_volume) AS max_bid_volume,
            approx_quantile(d_ask_volume, 0.99) AS p99_ask_volume,
            approx_quantile(d_bid_volume, 0.99) AS p99_bid_volume,
            count(*) FILTER (WHERE date_diff('millisecond', previous, t) >= 60000) AS gaps_ge_1min,
            max(date_diff('millisecond',previous,t)) AS max_gap_ms
        FROM sequenced;`, cwd);
    if (summary.ticks !== expected || summary.rows_with_nulls || summary.invalid_timestamps || summary.invalid_numbers) {
        throw new Error(`Validation du format échouée : ${JSON.stringify(summary)}`);
    }
    const coverage = duckdbJson(binary, `SET TimeZone='UTC';
        SELECT substr(timestamp_utc,1,16) AS minute_utc, count(*) AS ticks FROM ${scan}
        GROUP BY 1 ORDER BY 1;`, cwd);
    const hours = {};
    for (const row of coverage) {
        const hour = row.minute_utc.slice(0, 13);
        hours[hour] = (hours[hour] ?? 0) + row.ticks;
    }
    return {summary, hours, minutes: coverage};
}

function sha256(data) {
    return createHash('sha256').update(data).digest('hex');
}

async function main() {
    const {values} = parseArgs({options: {from: {type:'string'}, through: {type:'string'},
        out: {type:'string'}, duckdb: {type:'string'}}});
    if (!values.from || !values.through || !values.out || !values.duckdb) {
        throw new Error('Paramètres requis : --from AAAA-MM-JJ --through AAAA-MM-JJ --out DOSSIER --duckdb EXE');
    }
    const from = day(values.from);
    const through = day(values.through);
    if (from > through || through + dayMs > Date.now()) throw new Error('Sélectionner des journées UTC terminées.');
    const out = path.resolve(values.out);
    const binary = path.resolve(values.duckdb);
    const cache = path.join(out, 'cache');
    const log = path.join(out, 'journal_collecte.ndjson');
    const policy = {batchSize: 4, pauseBetweenBatchesMs: 2000, retryCount: 2,
        pauseBetweenRetriesMs: 1500, failAfterRetryCount: true, rateLimitRetryCount: 2,
        useCache: true, cacheFolderPath: cache};
    await fs.mkdir(cache, {recursive:true});
    const [duckdb] = duckdbJson(binary, 'SELECT version() AS version;', out);
    for (let stamp = from; stamp <= through; stamp += dayMs) {
        const first = new Date(stamp).toISOString();
        const last = new Date(stamp + dayMs).toISOString();
        const date = first.slice(0, 10);
        const folder = path.join(out, date.slice(0, 4), date.slice(5, 7));
        const file = path.join(folder, `${date}.csv`);
        const manifest = path.join(folder, `${date}.json`);
        const pending = path.join(folder, `${date}.part.csv`);
        const provenance = {date, from_utc: first, to_exclusive_utc: last, instrument: 'xauusd',
            source: 'Dukascopy', endpoint: URL_ROOT, dukascopy_node: libraryVersion,
            node: process.version, duckdb: duckdb.version, volume_units: 'millions', policy};
        const note = async details => {
            await fs.appendFile(log, JSON.stringify({...provenance, at_utc: new Date().toISOString(), ...details}) + '\n');
            console.log(date, details.status, details.ticks ?? details.message ?? '');
        };
        if (await fs.stat(manifest).catch(error => { if (error.code !== 'ENOENT') throw error; })) {
            const saved = JSON.parse(await fs.readFile(manifest, 'utf8'));
            if (saved.sha256 !== sha256(await fs.readFile(file))) throw new Error(`SHA-256 incompatible : ${file}`);
            console.log(date, 'sha256_verified_resume');
            continue;
        }
        await fs.mkdir(folder, {recursive:true});
        try {
            for (const artifact of [file, pending]) {
                if (await fs.stat(artifact).catch(error => { if (error.code !== 'ENOENT') throw error; })) {
                    throw new Error(`Lot local sans manifeste, conserver et examiner : ${artifact}`);
                }
            }
            const ticks = await getHistoricalRates({instrument:'xauusd', dates:{from:first,to:last},
                timeframe:'tick', format:'array', volumes:true, volumeUnits:'millions', ...policy});
            if (!ticks.length) {
                await note({status:'empty_unverified'});
                continue;
            }
            const csv = toCsv(ticks, stamp, stamp + dayMs);
            await fs.writeFile(pending, csv, {flag:'wx'});
            const validation = validate(binary, pending, ticks.length, out);
            const record = {...provenance, at_utc:new Date().toISOString(), sha256:sha256(csv),
                bytes:Buffer.byteLength(csv), ticks:ticks.length, validation};
            // Link without replacing any file that could already exist.
            await fs.link(pending, file);
            await fs.writeFile(manifest, JSON.stringify(record, null, 2) + '\n', {flag:'wx'});
            await fs.unlink(pending);
            await note({status:'collected_and_profiled', ticks:ticks.length, sha256:record.sha256});
        } catch (error) {
            await note({status:'failed', type:error.name, message:error.message});
            throw error; // Cached requests remain reusable. In particular, stop on denied/rate-limited access.
        }
    }
}

module.exports = {day, decimal, toCsv, validate};
if (require.main === module) main().catch(error => { console.error(error.name, error.message); process.exitCode = 1; });
