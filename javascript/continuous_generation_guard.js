(() => {
    "use strict";

    if (window.__forgeNeoAutoVramPurgeContinuousGuardInstalled) {
        return;
    }
    window.__forgeNeoAutoVramPurgeContinuousGuardInstalled = true;

    const endpoint = "/forge-neo-auto-vram-purge/continuous";
    const heartbeatMs = 2000;

    let continuousActive = false;
    let heartbeatTimer = null;

    function sendState(active, keepalive = false) {
        const url = `${endpoint}?active=${active ? "true" : "false"}&t=${Date.now()}`;
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

    document.addEventListener(
        "click",
        (event) => {
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
        },
        true,
    );

    window.addEventListener("beforeunload", () => {
        if (!continuousActive) {
            return;
        }

        if (heartbeatTimer !== null) {
            clearInterval(heartbeatTimer);
            heartbeatTimer = null;
        }

        try {
            navigator.sendBeacon(`${endpoint}?active=false`);
        } catch (_) {
            sendState(false, true);
        }
    });
})();
