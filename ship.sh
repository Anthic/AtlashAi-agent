#!/bin/bash
set -e

MSG="$1"
if [ -z "$MSG" ]; then
  echo "Error: Please provide a commit message."
  echo "Usage: ./ship.sh \"your commit message\""
  exit 1
fi

echo "[1/6] Running quick unit tests..."
python testing/test_pipeline.py

CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

echo "[2/6] Staging and committing changes..."
git add .
git commit -m "$MSG"

echo "[3/6] Pushing feature branch to GitHub..."
git push origin "$CURRENT_BRANCH"

if [ "$CURRENT_BRANCH" != "main" ]; then
  echo "[4/6] Switching to main & pulling latest..."
  git checkout main
  git pull origin main

  echo "[5/6] Merging $CURRENT_BRANCH into main..."
  git merge "$CURRENT_BRANCH"

  echo "[6/6] Pushing to main (Triggering GitHub Actions CI/CD)..."
  git push origin main

  echo "Returning to feature branch: $CURRENT_BRANCH"
  git checkout "$CURRENT_BRANCH"
fi

echo "Done! Code pushed & AWS Deployment triggered automatically!"
