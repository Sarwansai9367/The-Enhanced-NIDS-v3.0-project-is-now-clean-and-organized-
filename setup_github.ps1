# GitHub Setup Script for Enhanced NIDS v3.0
# This script will help you push your project to GitHub

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  🚀 GitHub Setup - Enhanced NIDS v3.0" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green

# Check if Git is installed
Write-Host "📋 Step 1: Checking Git installation..." -ForegroundColor Cyan

$gitInstalled = Get-Command git -ErrorAction SilentlyContinue

if ($gitInstalled) {
    $gitVersion = git --version
    Write-Host "  ✅ Git is installed: $gitVersion`n" -ForegroundColor Green
} else {
    Write-Host "  ❌ Git is not installed`n" -ForegroundColor Red
    Write-Host "Options to install Git:" -ForegroundColor Yellow
    Write-Host "`n1. Download Git for Windows:" -ForegroundColor White
    Write-Host "   https://git-scm.com/download/win`n" -ForegroundColor Cyan
    Write-Host "2. Or install via winget:" -ForegroundColor White
    Write-Host "   winget install --id Git.Git -e --source winget`n" -ForegroundColor Cyan
    Write-Host "After installing Git, run this script again!`n" -ForegroundColor Yellow

    $install = Read-Host "Would you like to try installing Git via winget now? (y/n)"

    if ($install -eq 'y' -or $install -eq 'Y') {
        Write-Host "`nInstalling Git via winget..." -ForegroundColor Yellow
        winget install --id Git.Git -e --source winget
        Write-Host "`n✅ Git installed! Please restart PowerShell and run this script again." -ForegroundColor Green
    }

    exit
}

# Git is installed, continue setup
Write-Host "📋 Step 2: Git Configuration" -ForegroundColor Cyan

# Check if Git is configured
$gitName = git config --global user.name
$gitEmail = git config --global user.email

if ([string]::IsNullOrWhiteSpace($gitName)) {
    Write-Host "  Git username not configured" -ForegroundColor Yellow
    $name = Read-Host "  Enter your name (for commits)"
    git config --global user.name "$name"
    Write-Host "  ✅ Name set: $name" -ForegroundColor Green
} else {
    Write-Host "  ✅ Git username: $gitName" -ForegroundColor Green
}

if ([string]::IsNullOrWhiteSpace($gitEmail)) {
    Write-Host "  Git email not configured" -ForegroundColor Yellow
    $email = Read-Host "  Enter your email (use your GitHub email)"
    git config --global user.email "$email"
    Write-Host "  ✅ Email set: $email" -ForegroundColor Green
} else {
    Write-Host "  ✅ Git email: $gitEmail" -ForegroundColor Green
}

Write-Host ""

# Check if already a Git repository
Write-Host "📋 Step 3: Initialize Git Repository" -ForegroundColor Cyan

if (Test-Path ".git") {
    Write-Host "  ✅ Already a Git repository`n" -ForegroundColor Green
} else {
    Write-Host "  Initializing Git repository..." -ForegroundColor Yellow
    git init
    Write-Host "  ✅ Git repository initialized`n" -ForegroundColor Green
}

# Clean the project first
Write-Host "📋 Step 4: Cleaning Project" -ForegroundColor Cyan
Write-Host "  Running cleanup.py..." -ForegroundColor Yellow

if (Test-Path "cleanup.py") {
    python cleanup.py | Out-Null
    Write-Host "  ✅ Project cleaned`n" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  cleanup.py not found, skipping...`n" -ForegroundColor Yellow
}

# Stage all files
Write-Host "📋 Step 5: Staging Files" -ForegroundColor Cyan
Write-Host "  Adding all files to Git..." -ForegroundColor Yellow

git add .

$stagedFiles = (git diff --cached --name-only | Measure-Object).Count
Write-Host "  ✅ Staged $stagedFiles files`n" -ForegroundColor Green

# Create initial commit
Write-Host "📋 Step 6: Creating Initial Commit" -ForegroundColor Cyan

$hasCommits = git log --oneline 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Creating first commit..." -ForegroundColor Yellow
    git commit -m "Initial commit: Enhanced NIDS v3.0 Enterprise Edition - Complete project with all features, documentation, and tests"
    Write-Host "  ✅ Initial commit created`n" -ForegroundColor Green
} else {
    Write-Host "  Repository already has commits" -ForegroundColor Yellow
    $createCommit = Read-Host "  Create a new commit with current changes? (y/n)"
    if ($createCommit -eq 'y' -or $createCommit -eq 'Y') {
        $message = Read-Host "  Enter commit message"
        git commit -m "$message"
        Write-Host "  ✅ Commit created`n" -ForegroundColor Green
    } else {
        Write-Host "  Skipping commit`n" -ForegroundColor Yellow
    }
}

# Set main branch
git branch -M main 2>$null

# GitHub repository setup
Write-Host "📋 Step 7: GitHub Repository" -ForegroundColor Cyan
Write-Host "`nBefore proceeding, you need to create a GitHub repository:" -ForegroundColor Yellow
Write-Host "`n1. Go to: https://github.com/new" -ForegroundColor White
Write-Host "2. Repository name: enhanced-nids (or your choice)" -ForegroundColor White
Write-Host "3. Description: Enhanced Network Intrusion Detection System v3.0" -ForegroundColor White
Write-Host "4. Choose Public or Private" -ForegroundColor White
Write-Host "5. DON'T initialize with README (we have one)" -ForegroundColor White
Write-Host "6. Click 'Create repository'`n" -ForegroundColor White

$repoCreated = Read-Host "Have you created the GitHub repository? (y/n)"

if ($repoCreated -eq 'y' -or $repoCreated -eq 'Y') {
    Write-Host ""
    $username = Read-Host "Enter your GitHub username"
    $repoName = Read-Host "Enter repository name (default: enhanced-nids)"

    if ([string]::IsNullOrWhiteSpace($repoName)) {
        $repoName = "enhanced-nids"
    }

    $repoUrl = "https://github.com/$username/$repoName.git"

    Write-Host "`nRepository URL: $repoUrl" -ForegroundColor Cyan

    # Check if remote already exists
    $remotes = git remote
    if ($remotes -contains "origin") {
        Write-Host "  Remote 'origin' already exists" -ForegroundColor Yellow
        $updateRemote = Read-Host "  Update remote URL? (y/n)"
        if ($updateRemote -eq 'y' -or $updateRemote -eq 'Y') {
            git remote set-url origin $repoUrl
            Write-Host "  ✅ Remote URL updated`n" -ForegroundColor Green
        }
    } else {
        git remote add origin $repoUrl
        Write-Host "  ✅ Remote 'origin' added`n" -ForegroundColor Green
    }

    # Push to GitHub
    Write-Host "📋 Step 8: Pushing to GitHub" -ForegroundColor Cyan
    Write-Host "  Pushing to GitHub..." -ForegroundColor Yellow
    Write-Host "  (You may be prompted to sign in to GitHub)`n" -ForegroundColor White

    $pushResult = git push -u origin main 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n" + "="*70 -ForegroundColor Green
        Write-Host "  🎉 SUCCESS! Project pushed to GitHub!" -ForegroundColor Green
        Write-Host "="*70 -ForegroundColor Green
        Write-Host "`nYour repository is now at:" -ForegroundColor Cyan
        Write-Host "  https://github.com/$username/$repoName`n" -ForegroundColor Yellow

        Write-Host "Next steps:" -ForegroundColor Cyan
        Write-Host "  1. Visit your repository on GitHub" -ForegroundColor White
        Write-Host "  2. Add topics/tags for discoverability" -ForegroundColor White
        Write-Host "  3. Star your own repository ⭐" -ForegroundColor White
        Write-Host "  4. Share the link!`n" -ForegroundColor White

    } else {
        Write-Host "`n❌ Push failed. Error:" -ForegroundColor Red
        Write-Host $pushResult -ForegroundColor Yellow
        Write-Host "`nTroubleshooting tips:" -ForegroundColor Cyan
        Write-Host "  1. Make sure you're signed in to GitHub" -ForegroundColor White
        Write-Host "  2. Check repository URL is correct" -ForegroundColor White
        Write-Host "  3. Try using GitHub Desktop instead" -ForegroundColor White
        Write-Host "  4. See GITHUB_GUIDE.md for more help`n" -ForegroundColor White
    }

} else {
    Write-Host "`nNo problem! Here's what to do:" -ForegroundColor Yellow
    Write-Host "`n1. Create repository on GitHub:" -ForegroundColor White
    Write-Host "   https://github.com/new" -ForegroundColor Cyan
    Write-Host "`n2. Then run this script again!`n" -ForegroundColor White
}

Write-Host "For detailed instructions, see: GITHUB_GUIDE.md`n" -ForegroundColor Cyan

Write-Host "========================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green
