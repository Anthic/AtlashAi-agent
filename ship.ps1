param (
    [Parameter(Mandatory=$true)]
    [string]$msg
)

Write-Host "[1/6] Running quick unit tests..." -ForegroundColor Cyan
python testing/test_pipeline.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Tests failed! Aborting push to prevent broken deploy." -ForegroundColor Red
    exit 1
}

$currentBranch = (git branch --show-current).Trim()
Write-Host "Current branch: $currentBranch" -ForegroundColor Yellow

Write-Host "[2/6] Staging and committing changes..." -ForegroundColor Cyan
git add .
git commit -m "$msg"

Write-Host "⬆[3/6] Pushing feature branch to GitHub..." -ForegroundColor Cyan
git push origin $currentBranch

if ($currentBranch -ne "main") {
    Write-Host "[4/6] Switching to main & pulling latest..." -ForegroundColor Cyan
    git checkout main
    git pull origin main

    Write-Host "[5/6] Merging $currentBranch into main..." -ForegroundColor Cyan
    git merge $currentBranch

    Write-Host "[6/6] Pushing to main (Triggering GitHub Actions CI/CD)..." -ForegroundColor Green
    git push origin main

    Write-Host "Returning to feature branch: $currentBranch" -ForegroundColor Cyan
    git checkout $currentBranch
}

Write-Host "Done! Code pushed & AWS Deployment triggered automatically!" -ForegroundColor Green
