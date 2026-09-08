# PostgreSQL credentials

The PostgreSQL database scripts require a password supplied through `PGPASSWORD` or
the `--password` option. They do not contain a password fallback, and they fail before
opening a database connection when neither source is present.

```bash
PGPASSWORD='your-password' python scripts/db/db_create.py
PGPASSWORD='your-password' python scripts/db/upload_to_postgres.py
python scripts/db/db_create.py --password 'your-password'
python scripts/db/upload_to_postgres.py --password 'your-password'
```

The legacy `clinical_unified` export accepts an explicitly configured
`CLINICAL_DB_URL`; otherwise it requires `PGPASSWORD` before creating its engine.

The previously exposed hard-coded credential must be treated as compromised and rotated
externally in PostgreSQL and any secret store. This change does not perform credential
rotation.
