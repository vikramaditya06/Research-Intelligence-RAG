"""Initialize a development database using the same Alembic migrations as deployment."""
import subprocess
import sys

result = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=False)
if result.returncode != 0:
    raise SystemExit(result.returncode)
print("Database initialized with Alembic migrations.")
