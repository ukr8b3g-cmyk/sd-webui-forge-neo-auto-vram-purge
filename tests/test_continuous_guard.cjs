"use strict";
const assert = require("node:assert/strict");
const {readFileSync} = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {test} = require("node:test");

const source = readFileSync(path.join(__dirname, "../javascript/continuous_generation_guard.js"), "utf8");

function harness(id = "page-A", beacon = () => true) {
    const requests = [];
    const beacons = [];
    const intervals = new Map();
    const handlers = {};
    let nextTimer = 0;
    let click;
    const root = {addEventListener(event, fn) { click = fn; }};
    const window = {
        crypto: {randomUUID: () => id},
        addEventListener(event, fn) { handlers[event] = fn; },
    };
    const context = vm.createContext({
        window, document: root,
        gradioApp: () => root,
        onUiLoaded: fn => fn(),
        fetch(url, options) {
            requests.push({url, options});
            // Leave requests unresolved: ordering must not depend on completion.
            return new Promise(() => {});
        },
        navigator: {sendBeacon(url) { beacons.push(url); return beacon(url); }},
        setInterval(fn) { const id = ++nextTimer; intervals.set(id, fn); return id; },
        clearInterval(id) { intervals.delete(id); },
        Date, Math, encodeURIComponent,
    });
    vm.runInContext(source, context);
    return {
        requests, beacons, intervals, handlers, context,
        click(label) { click({target: {closest: () => ({textContent: label})}}); },
        heartbeat() { for (const fn of [...intervals.values()]) fn(); },
        reloadScript() { vm.runInContext(source, context); },
    };
}

function state(url) {
    const params = new URL(url, "http://forge.test").searchParams;
    return {id: params.get("client_id"), seq: Number(params.get("sequence")),
            active: params.get("active")};
}

test("ON, heartbeat and OFF share one page ID and monotonically increasing sequence", () => {
    const h = harness();
    h.click("Generate forever");
    h.heartbeat();
    h.click("Cancel generate forever");
    assert.deepEqual(h.requests.map(r => state(r.url)), [
        {id: "page-A", seq: 1, active: "true"},
        {id: "page-A", seq: 2, active: "true"},
        {id: "page-A", seq: 3, active: "false"},
    ]);
    assert.equal(h.intervals.size, 0);
    assert.ok(h.requests.every(r => r.options.method === "POST"
        && r.options.credentials === "same-origin"));
});

test("separate tabs use separate page IDs and independent sequence counters", () => {
    const a = harness("tab-A");
    const b = harness("tab-B");
    a.click("Generate forever");
    a.heartbeat();
    b.click("Generate forever");
    b.click("Cancel generate forever");
    assert.equal(state(a.requests.at(-1).url).id, "tab-A");
    assert.deepEqual(state(b.requests.at(-1).url), {id: "tab-B", seq: 2, active: "false"});
    assert.equal(a.intervals.size, 1);
});

test("repeated ON leaves a single heartbeat timer; duplicate script is ignored", () => {
    const h = harness();
    h.click("Generate forever");
    h.click("Generate forever");
    h.reloadScript();
    assert.equal(h.intervals.size, 1);
    h.heartbeat();
    assert.equal(h.requests.length, 3);
    assert.equal(state(h.requests[2].url).seq, 3);
});

test("page close sends an ordered OFF beacon and stops heartbeat", () => {
    const h = harness();
    h.click("Generate forever");
    h.handlers.beforeunload();
    assert.deepEqual(state(h.beacons[0]), {id: "page-A", seq: 2, active: "false"});
    assert.equal(h.intervals.size, 0);
    assert.equal(h.requests.length, 1);
});

test("sendBeacon false falls back to keepalive with the same OFF sequence", () => {
    const h = harness("page-A", () => false);
    h.click("Generate forever");
    h.handlers.beforeunload();
    assert.equal(h.requests[1].url, h.beacons[0]);
    assert.equal(h.requests[1].options.keepalive, true);
    assert.equal(state(h.requests[1].url).seq, 2);
});

test("sendBeacon exception falls back to a newer ordered keepalive OFF", () => {
    const h = harness("page-A", () => { throw Error("unsupported"); });
    h.click("Generate forever");
    h.handlers.beforeunload();
    assert.deepEqual(state(h.requests[1].url), {id: "page-A", seq: 3, active: "false"});
    assert.equal(h.requests[1].options.keepalive, true);
});

test("page with no continuous loop sends no close notification", () => {
    const h = harness();
    h.handlers.beforeunload();
    assert.equal(h.requests.length, 0);
    assert.equal(h.beacons.length, 0);
});

test("ID fallback works when crypto.randomUUID is unavailable", () => {
    const h = harness();
    // Evaluate a fresh page in the same sandbox, without the installed guard.
    delete h.context.window.__forgeNeoAutoVramPurgeContinuousGuardInstalled;
    delete h.context.window.crypto;
    h.reloadScript();
    h.click("Generate forever");
    assert.ok(state(h.requests[0].url).id.length > 0);
    assert.notEqual(state(h.requests[0].url).id, "page-A");
});
