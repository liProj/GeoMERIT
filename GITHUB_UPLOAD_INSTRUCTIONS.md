# GitHub Upload Instructions

The repository is already initialized locally and committed on branch `main`.

Local repository path:

```text
H:\胜利油田新实验\GeoMERIT-GitHub-Release
```

Current commit:

```bash
git log --oneline -1
```

## Option 1: Create the GitHub repository in the browser

1. Open GitHub and create a new empty repository, for example:
   `GeoMERIT`
2. Do not initialize it with README, license, or `.gitignore`, because this
   local repository already has those files.
3. Push:

```bash
cd /d H:\胜利油田新实验\GeoMERIT-GitHub-Release
git remote add origin https://github.com/<your-username>/GeoMERIT.git
git push -u origin main
```

## Option 2: Use a GitHub personal access token

If a token is available, create the remote repository with the GitHub API:

```powershell
$env:GITHUB_TOKEN = "<your-token>"
curl.exe -H "Authorization: Bearer $env:GITHUB_TOKEN" `
  -H "Accept: application/vnd.github+json" `
  https://api.github.com/user/repos `
  -d "{\"name\":\"GeoMERIT\",\"private\":false,\"description\":\"Missingness-aware and penalty-guided robust lithology prediction from well logs\"}"
```

Then push:

```bash
git remote add origin https://github.com/<your-username>/GeoMERIT.git
git push -u origin main
```

## After Upload

Update `CITATION.cff`:

```yaml
repository-code: "https://github.com/<your-username>/GeoMERIT"
```

Then commit and push:

```bash
git add CITATION.cff
git commit -m "Update repository citation URL"
git push
```

