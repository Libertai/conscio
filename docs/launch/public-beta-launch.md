# Conscio Public Beta Launch Plan

This launch is a public beta for operators and researchers who can run a
dedicated VM, inspect logs, and tolerate sharp edges. It is not a hosted
consumer product and not an unattended production SRE promise.

## Launch Objective

Put Conscio in front of serious users as a real long-running agent harness:
persistent memory, autonomous heartbeats, operator controls, audited tool
execution, prompt-injection-aware memory writes, and a documented deployment
path.

## Launch Criteria

- The public beta branch is pushed and tagged.
- `scripts/check-launch-readiness.sh` passes locally.
- The reference VM passes `conscio service doctor`.
- `GET /health`, `GET /ui`, authenticated `GET /status`, and authenticated
  `GET /metrics` pass on the reference VM.
- DB backup exists and restore instructions are tested enough to be credible.
- README points to public-beta docs.
- Known limits are published, not hidden.
- Rollback command is known before announcing.

## Launch Sequence

1. Freeze the launch branch.
2. Run local readiness:

   ```bash
   scripts/check-launch-readiness.sh
   ```

3. Run VM readiness:

   ```bash
   CONSCIO_LAUNCH_URL=https://grit-carry-state-false.2n6.me \
   CONSCIO_LAUNCH_SSH_HOST=root@2a01:e0a:ff0:3d41:3:66a4:5f53:2921 \
   scripts/check-launch-readiness.sh --remote
   ```

4. Create a Git tag:

   ```bash
   git tag public-beta-YYYYMMDD
   git push origin public-beta-YYYYMMDD
   ```

5. Publish the announcement.
6. Monitor `/metrics`, service journal, tool events, and UI auth errors for the
   first hour.

## Rollback

Use the previous known-good commit or tag. On the reference VM:

```bash
systemctl stop conscio
cd /opt/conscio
git fetch origin
git reset --hard <known-good-ref>
chown -R conscio:conscio /opt/conscio
su -s /bin/bash conscio -c "cd /opt/conscio && .venv/bin/pip install -e ."
systemctl start conscio
```

The pre-launch state backup should stay available until the launch has survived
at least one full autonomous soak period.

## First-Hour Watch

- Service remains active under systemd.
- `paused` remains false unless intentionally paused.
- `last_error` is empty or understood.
- Tool events are being recorded and secrets are redacted.
- No repeated restart loop in `journalctl -u conscio`.
- UI loads the current static bundle after hard refresh.

## Launch Owner Checklist

- [ ] Local readiness passed.
- [ ] VM readiness passed.
- [ ] Backup path recorded.
- [ ] Tag pushed.
- [ ] Announcement published.
- [ ] First-hour watch completed.
- [ ] Known issues updated from launch observations.
