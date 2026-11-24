const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');

const [collector, folder, duckdb] = process.argv.slice(2);
const {validate} = require(path.resolve(collector));
const attempted = new Map();
const journal = fs.readFileSync(path.join(folder, 'heures.tsv'), 'utf8').trimEnd().split(/\r?\n/).slice(1);
for (const line of journal) {
    const [date, hour, status, textCount] = line.split('\t');
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || !/^\d{1,2}$/.test(hour) || !/^\d+$/.test(textCount)) {
        throw new Error('Invalid hourly manifest');
    }
    if (!attempted.has(date)) attempted.set(date, []);
    attempted.get(date).push({hour: Number(hour), status, ticks: Number(textCount)});
}

for (const [date, hours] of attempted) {
    const csv = path.join(folder, `${date}.csv`);
    if (!fs.existsSync(csv)) continue;
    const manifest = path.join(folder, `${date}.audit.json`);
    if (fs.existsSync(manifest)) continue;
    const raw = fs.readFileSync(csv);
    const ticks = hours.reduce((sum, hour) => sum + hour.ticks, 0);
    const sha256 = createHash('sha256').update(raw).digest('hex');
    const result = {date, source: 'JForex DEMO SDK 3.6.51 / IHistory.getTicks', sha256, bytes: raw.length,
        status: ticks ? 'audited' : 'empty_unverified', validation_status: 'pending', requested_hours: hours};
    if (ticks) {
        Object.assign(result, validate(duckdb, csv, ticks, folder));
        if (result.summary.ticks !== ticks) throw new Error(`Audit count does not match hour manifest: ${date}`);
    } else if (raw.toString().trimEnd().includes('\n')) {
        throw new Error(`Empty hour manifest conflicts with CSV: ${date}`);
    }
    fs.writeFileSync(manifest, JSON.stringify(result, null, 2) + '\n', {flag: 'wx'});
    console.log(date, result.status, ticks);
}
