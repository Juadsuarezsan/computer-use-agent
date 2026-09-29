# Sandbox image: Ubuntu 22.04 desktop with a virtual X display, the input /
# screenshot tools used by XdoToolVM, a VNC server + noVNC for the live viewer,
# and the applications the tasks operate (Firefox, LibreOffice).
#
# Validation of this image needs a Docker daemon and is pending (see
# docs/architecture.md, "Infrastructure status").
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:1 \
    DISPLAY_WIDTH=1024 \
    DISPLAY_HEIGHT=768 \
    VNC_PORT=5900 \
    NOVNC_PORT=6080

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        xvfb x11vnc xdotool scrot novnc websockify \
        xterm fluxbox dbus-x11 \
        firefox libreoffice-calc libreoffice-writer \
        fonts-dejavu-core ca-certificates curl python3 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/sandbox-entrypoint.sh /usr/local/bin/sandbox-entrypoint.sh
COPY sandbox/webapps /srv/webapps
RUN chmod +x /usr/local/bin/sandbox-entrypoint.sh

EXPOSE 5900 6080 8080
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s \
    CMD xdotool getdisplaygeometry >/dev/null 2>&1 || exit 1
ENTRYPOINT ["/usr/local/bin/sandbox-entrypoint.sh"]
