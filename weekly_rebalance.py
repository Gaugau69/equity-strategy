"""
weekly_rebalance.py
-------------------
Script tout-en-un pour le rebalancement hebdomadaire.

Lance chaque lundi matin :
    python weekly_rebalance.py

Ou pour tester sans passer d'ordres :
    python weekly_rebalance.py --dry-run

Ce script :
  1. Met à jour les données S&P 500 via Yahoo Finance
  2. Génère les signaux via notre pipeline
  3. Passe les ordres sur IBKR paper trading
  4. Envoie un résumé dans outputs/paper_trading/weekly_summary.txt
"""

import sys
import subprocess

# Fix Unicode pour terminal Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import argparse
import logging
from pathlib import Path
from datetime import date

LOG_DIR = Path("outputs/paper_trading")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "weekly_rebalance.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


def run(cmd: list[str], step: str) -> bool:
    """Run a subprocess command, return True if successful."""
    log.info(f"\n{'='*55}")
    log.info(f"  {step}")
    log.info(f"{'='*55}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        log.error(f"  FAILED (exit code {result.returncode})")
        return False
    log.info(f"  OK")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Génère les signaux mais ne passe pas d'ordres")
    parser.add_argument("--force", action="store_true",
                        help="Force le run même si pas lundi")
    parser.add_argument("--skip-update", action="store_true",
                        help="Saute la mise à jour des données")
    args = parser.parse_args()

    today = date.today()
    if today.weekday() != 0 and not args.force:
        log.info(f"Aujourd'hui c'est {today.strftime('%A')} — pas lundi.")
        log.info("Utilise --force pour forcer le run.")
        return

    log.info("\n" + "="*55)
    log.info("  WEEKLY REBALANCE")
    log.info(f"  {today.strftime('%A %d %B %Y')}")
    log.info(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE PAPER'}")
    log.info("="*55)

    # ── Étape 1 : mise à jour des données ─────────────────────────────────────
    if not args.skip_update:
        ok = run(
            [sys.executable, "data_ingestion/data_collection.py"],
            "Etape 1/3 - Mise a jour des donnees S&P 500"
        )
        if not ok:
            log.warning("  Mise a jour echouee — on continue avec les donnees existantes")
    else:
        log.info("\nEtape 1/3 — Mise a jour des donnees : IGNOREE (--skip-update)")

    # ── Étape 2 & 3 : signaux + ordres ────────────────────────────────────────
    executor_args = [sys.executable, "ibkr_executor.py", "--force"]
    if args.dry_run:
        executor_args.append("--dry-run")

    ok = run(executor_args, "Etapes 2+3/3 - Signaux + Ordres IBKR")
    if not ok:
        log.error("  Execution echouee")
        return

    log.info("\n" + "="*55)
    log.info("  REBALANCEMENT TERMINE")
    if args.dry_run:
        log.info("  Mode DRY RUN — aucun ordre passe")
    else:
        log.info("  Ordres passes sur IBKR paper trading")
    log.info(f"  Logs : {LOG_DIR}/")
    log.info("="*55 + "\n")


if __name__ == "__main__":
    main()