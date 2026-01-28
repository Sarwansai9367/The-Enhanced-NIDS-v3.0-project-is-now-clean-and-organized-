# 🔄 Quick Git Commands Reference

## ✅ Your project is on GitHub!

**Repository:** https://github.com/Sarwansai9367/The-Enhanced-NIDS-v3.0-project-is-now-clean-and-organized-

---

## 🚀 **Making Updates to Your Project**

### Every Time You Make Changes:

```powershell
# Step 1: Make sure Git works (run once per PowerShell session)
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Or just restart PowerShell - Git will work automatically!

# Step 2: Clean your project
python cleanup.py

# Step 3: Check what changed
git status

# Step 4: Add all changes
git add .

# Step 5: Commit with a message
git commit -m "Your description of what you changed"

# Step 6: Push to GitHub
git push
```

---

## 📝 **Common Git Commands**

### View Changes
```powershell
git status                    # See what files changed
git diff                      # See detailed changes
git log                       # See commit history
```

### Commit Changes
```powershell
git add .                     # Stage all changes
git add filename.py           # Stage specific file
git commit -m "message"       # Commit with message
```

### Push/Pull
```powershell
git push                      # Upload to GitHub
git pull                      # Download from GitHub
```

### Undo Changes
```powershell
git checkout -- filename.py   # Undo changes to one file
git reset HEAD~1              # Undo last commit (keep changes)
git reset --hard HEAD~1       # Undo last commit (delete changes)
```

---

## 🔧 **Troubleshooting**

### "git: command not found"

**Solution 1: Refresh PATH**
```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
```

**Solution 2: Restart PowerShell**
- Close and reopen PowerShell
- Git will work automatically

### "Permission denied" or "Authentication failed"

**Solution:** You may need to sign in again
- GitHub will prompt you
- Sign in through the browser window

---

## 📌 **Quick Workflow**

### Daily Development Workflow:

```powershell
# Morning - Start work
git pull                      # Get latest changes

# ... make changes to your code ...

# Evening - Save work
python cleanup.py             # Clean project
git add .                     # Stage changes
git commit -m "What I did today"
git push                      # Upload to GitHub
```

---

## 🎯 **Example Commit Messages**

Good commit messages:
- ✅ "Add deep learning LSTM detector"
- ✅ "Fix: SQLite syntax error in logger"
- ✅ "Update documentation for deployment"
- ✅ "Improve performance of packet capture"

Bad commit messages:
- ❌ "update"
- ❌ "fix bug"
- ❌ "changes"

---

## 🌐 **Your Repository Info**

**Name:** The-Enhanced-NIDS-v3.0-project-is-now-clean-and-organized-  
**Owner:** Sarwansai9367  
**URL:** https://github.com/Sarwansai9367/The-Enhanced-NIDS-v3.0-project-is-now-clean-and-organized-  
**Branch:** main  

---

## 📱 **GitHub Mobile**

Download GitHub mobile app to:
- View your repository
- Monitor activity
- Respond to issues
- Update code on the go

**Download:** https://github.com/mobile

---

## ✅ **Quick Checklist**

Before pushing to GitHub:
- [ ] Clean project (`python cleanup.py`)
- [ ] Test code (`python test_complete.py`)
- [ ] Write good commit message
- [ ] Check what's being pushed (`git status`)
- [ ] Push to GitHub (`git push`)

---

## 💡 **Pro Tips**

1. **Commit often** - Small, frequent commits are better
2. **Write clear messages** - Future you will thank you
3. **Clean before committing** - Run `cleanup.py` first
4. **Test before pushing** - Make sure code works
5. **Pull before pushing** - Get latest changes first

---

## 🎓 **Learning Resources**

- **Git Tutorial:** https://git-scm.com/book
- **GitHub Guides:** https://guides.github.com
- **Git Cheat Sheet:** https://education.github.com/git-cheat-sheet-education.pdf

---

**Your repository:** https://github.com/Sarwansai9367/The-Enhanced-NIDS-v3.0-project-is-now-clean-and-organized-

**Keep coding and pushing updates!** 🚀
