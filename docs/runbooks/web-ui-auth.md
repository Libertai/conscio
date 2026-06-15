# Web UI Auth

## Symptoms

- `/ui` redirects to login repeatedly.
- Login returns 401 or 429.
- Browser works over localhost but fails behind HTTPS.

## Check

Verify `web_password`:

```bash
grep -n "web_password" /home/conscio/.conscio/config.toml
```

For HTTPS deployments, verify:

```toml
[service]
web_secure_cookies = true
```

For localhost-only development, `web_secure_cookies = false` is acceptable.

## Recover

Set a new strong `web_password`, restart the service, and clear the browser
cookie for the host:

```bash
conscio service stop
conscio service start
```

If rate limited by failed logins, wait for the failure window to expire or
restart the service after confirming the password.
