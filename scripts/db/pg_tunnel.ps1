# Opens the SSH tunnel to the aecd_platform PostgreSQL on the storage-LAN server.
#
# WHY: since 2026-10-02 aecd_platform lives on the HPE ML30 server
#      (192.168.10.133, Docker container sers-db). Its port 5432 is bound to
#      the server's 127.0.0.1 only and ufw allows nothing but SSH, so every
#      client reaches it through this tunnel. With WSL mirrored networking the
#      same localhost:15432 works from Windows (Power BI) and WSL (scripts).
#
# Usage (PowerShell, Windows side — uses the Windows OpenSSH key):
#   powershell -ExecutionPolicy Bypass -File scripts\db\pg_tunnel.ps1
# Leave it running; it reconnects if the link drops. Ctrl+C to stop.

param(
    [string]$Server = "sysadmin@192.168.10.133",
    [int]$LocalPort = 15432
)

while ($true) {
    ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes `
        -o ServerAliveInterval=30 -o ServerAliveCountMax=3 `
        -L "127.0.0.1:${LocalPort}:127.0.0.1:5432" $Server
    Write-Host "tunnel closed (exit $LASTEXITCODE) - retrying in 10s"
    Start-Sleep -Seconds 10
}
