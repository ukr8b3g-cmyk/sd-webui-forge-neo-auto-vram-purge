(() => {
    "use strict";

    if (window.__forgeNeoAutoVramPurgeContinuousGuardInstalled) {
        return;
    }
    window.__forgeNeoAutoVramPurgeContinuousGuardInstalled = true;

    const endpoint = "/forge-neo-auto-vram-purge/continuous";
    const heartbeatMs = 2000;
    // A new ID for every loaded page (including duplicated tabs). sessionStorage
    // can be copied when opening a tab, so it must not identify the sender.
    const clientId = typeof window.crypto?.randomUUID === "function"
        ? window.crypto.randomUUID()
        : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
    let sequence = 0;

    let continuousActive = false;
    let heartbeatTimer = null;
    let clickListenerInstalled = false;

    function stateUrl(active) {
        return `${endpoint}?active=${active ? "true" : "false"}`
            + `&client_id=${encodeURIComponent(clientId)}&sequence=${++sequence}`;
    }

    function sendState(active, keepalive = false, url = stateUrl(active)) {
        try {
            fetch(url, {
                method: "POST",
                credentials: "same-origin",
                cache: "no-store",
                keepalive: keepalive,
            }).catch(() => {});
        } catch (_) {
            // The backend TTL is the fail-safe if a transient request fails.
        }
    }

    function setContinuousActive(active) {
        continuousActive = active;

        if (heartbeatTimer !== null) {
            clearInterval(heartbeatTimer);
            heartbeatTimer = null;
        }

        sendState(active);

        if (active) {
            heartbeatTimer = setInterval(() => sendState(true), heartbeatMs);
        }
    }

    function handleContextMenuClick(event) {
        const target = event.target;
        const item = target && target.closest ? target.closest("#context-menu a") : null;
        if (!item) {
            return;
        }

        const label = (item.textContent || "").trim();
        if (label === "Generate forever") {
            setContinuousActive(true);
        } else if (label === "Cancel generate forever") {
            setContinuousActive(false);
        }
    }

    function installClickListener() {
        if (clickListenerInstalled) {
            return;
        }

        const root = typeof gradioApp === "function" ? gradioApp() : document;
        if (!root || !root.addEventListener) {
            return;
        }

        root.addEventListener("click", handleContextMenuClick, true);
        clickListenerInstalled = true;
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(installClickListener);
    } else {
        window.addEventListener("load", installClickListener, {once: true});
    }

    window.addEventListener("beforeunload", () => {
        if (!continuousActive) {
            return;
        }

        if (heartbeatTimer !== null) {
            clearInterval(heartbeatTimer);
            heartbeatTimer = null;
        }

        try {
            const url = stateUrl(false);
            if (!navigator.sendBeacon(url)) {
                sendState(false, true, url);
            }
        } catch (_) {
            sendState(false, true);
        }
    });
})();
